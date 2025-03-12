from dataclasses import dataclass, field
from datetime import datetime
import os
import time
from typing import Callable

import gymnasium as gym
import minigrid
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm


TF_TO_NP_DTYPE = {
    tf.float16: np.float16,
    tf.float32: np.float32,
    tf.float64: np.float64,
    tf.int16: np.int16,
    tf.int32: np.int32,
    tf.int64: np.int64,
    tf.uint8: np.uint8,
    tf.uint16: np.uint16,
    tf.bool: bool,
}


@dataclass
class PPOHparams:
    total_steps: int
    horizon: int
    num_envs: int
    epochs: int = 5
    learning_rate: float = 5e-4
    batch_size: int | None = None
    clip_value: float = 0.1
    gamma: float = 0.99
    gae_lambda:  float = 0.95
    critic_coef: float = 1
    entropy_coef: float = 0.01
    annealing_steps: int | None = None
    final_learning_rate: float | None = None
    clip_by_norm: float | None = None

    normalize_value: bool = False
    normalize_batch_advantage: bool = True

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = self.horizon * self.num_envs
        if (self.annealing_steps is not None) ^ (self.final_learning_rate is not None):
            raise ValueError("both annealing_steps and final_learning_rate must be provided")
        elif self.batch_size > (self.horizon * self.num_envs):
            raise ValueError("batch size cannot be bigger than the num_envs times the horizon")


def get_gae_estimator(gamma: float, gae_lambda: float):
    def estimate_gae(estimated_value, next_estimated_value, reward, terminal):
        delta = reward + (1 - terminal) * gamma * next_estimated_value - estimated_value
        gae = np.empty((len(delta), 1), dtype=np.float32)
        for i in range(len(gae) - 1, -1, -1):
            if i == (len(gae) - 1):
                gae[i] = delta[i]
            else:
                gae[i] = delta[i] + gae_lambda * gamma * (1 - terminal[i]) * gae[i + 1]
        return gae
    return estimate_gae


def vectorize_gae_estimator(gae_estimator):
    def estimate(estimated_value, next_estimated_value, reward, terminal):
        return np.concatenate(
            [
                np.expand_dims(gae_estimator(
                    estimated_value[:, i],
                    next_estimated_value[:, i],
                    reward[:, i],
                    terminal[:, i]
                    ), axis=1) for i in range(estimated_value.shape[1])
            ],
            axis=1
        )
    return estimate


def build_loss(policy: keras.Model, old_policy: keras.Model, config: PPOHparams, return_all: bool):
    clip_value, critic_coef, entropy_coef = config.clip_value, config.critic_coef, config.entropy_coef
    def ppo_loss(obs, actions, advantage, vtarget):
        """Expected shapes:
          * obs: (B, ...)
          * actions: (B, 1)
          * advantage: (B, 1)
          * vtarget: (B, 1)
          """
        logits, estimated_value = policy(obs)
        log_probs = logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True)
        ac_logprob = tf.gather_nd(
            log_probs,
            tf.stack([tf.range(len(log_probs)), tf.squeeze(actions)], axis=1)
        )  # shape [b]

        old_logits, _ = old_policy(obs)
        old_logits = tf.stop_gradient(old_logits)
        old_logprobs = old_logits - tf.math.reduce_logsumexp(old_logits, axis=-1, keepdims=True)
        old_ac_logprob = tf.gather_nd(
            old_logprobs,
            tf.stack([tf.range(len(old_logprobs)), tf.squeeze(actions)], axis=1)
        )  # shape [b]

        log_ratio = ac_logprob - old_ac_logprob
        ratio = tf.math.exp(log_ratio)
        
        cpi = ratio * tf.squeeze(advantage)
        
        clipped = tf.clip_by_value(
            ratio, 1 - clip_value, 1 + clip_value
        ) * tf.squeeze(advantage)
        
        loss_clip = -tf.math.reduce_mean(
            tf.math.reduce_min(
                tf.stack([cpi, clipped], axis=-1), axis=-1
            )
        )

        loss_critic = tf.math.reduce_mean(tf.math.square(vtarget - estimated_value))

        # Entropy computation
        probs = tf.math.exp(log_probs)
        # Adds epsilon to prevent log(0)
        probs = tf.clip_by_value(probs, 1e-15, 1.0)
        # The entropy has a negative sign... it is ommited because I want to maximize it
        loss_entropy =  tf.math.reduce_mean(
            tf.math.reduce_sum(probs * log_probs, axis=-1)
        )

        loss = loss_clip + critic_coef * loss_critic + entropy_coef * loss_entropy

        if return_all:
            #  Those quantities are for monitoring purposes
            # aprox KL
            #  TODO: look for a better estimator here: http://joschu.net/blog/kl-approx.html
            aprox_kl = tf.math.reduce_mean(-log_ratio)
            # clip fraction
            clip_fraction = tf.math.reduce_sum(
                tf.cast(
                    tf.logical_or(ratio < (1 - clip_value), ratio > (1 + clip_value)),
                    dtype=tf.float32
                )
            ) / tf.cast(tf.shape(ratio)[0], dtype=tf.float32)
            return loss, loss_clip, loss_critic, loss_entropy, aprox_kl, clip_fraction
        return loss
    return tf.function(ppo_loss)


def sample_logits(logits):
    # https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    u = tf.random.uniform(tf.shape(logits))
    return tf.expand_dims(
        tf.argmax(
            logits - tf.math.log(-tf.math.log(u)),
            axis=-1,
            output_type=tf.int32),
        axis=-1,
    )


def extract_obs_batch(batch_idx, obs):
    if isinstance(obs, tuple):
        return tuple([extract_obs_batch(batch_idx, ob) for ob in obs])
    return obs[batch_idx]


def old_policy_updater(policy, old_policy):
    def update_old_policy():
        for w_old, w in zip(old_policy.trainable_variables, policy.trainable_variables):
            w_old.assign(w)
    return update_old_policy


@dataclass
class Buffer:
    obs: np.ndarray | tuple[np.ndarray]
    action: np.ndarray
    reward: np.ndarray
    estimated_value: np.ndarray
    next_estimated_value: np.ndarray
    terminal: np.ndarray

    def __len__(self):
        return self._buffer_idx

    def __post_init__(self):
        self._buffer_idx = 0
        self._max_length = len(self.terminal)
        if isinstance(self.obs, tuple):
            self._feed_obs = self._feed_multi_obs
        else:
            self._feed_obs = self._feed_single_obs

    def _feed_multi_obs(self, obs: tuple[np.ndarray]):
        for buffer_ob, ob in zip(self.obs, obs):
            buffer_ob[self._buffer_idx] = ob

    def _feed_single_obs(self, obs: np.ndarray):
        self.obs[self._buffer_idx] = obs

    def add(self, obs, action, reward, value, next_value, terminal):
        if self._buffer_idx >=  self._max_length:
            raise RuntimeError("Buffer is full, it must be emptied before adding new data")
        self._feed_obs(obs)
        self.action[self._buffer_idx] = action
        self.reward[self._buffer_idx] = reward
        self.estimated_value[self._buffer_idx] = value
        self.next_estimated_value[self._buffer_idx] = next_value
        self.terminal[self._buffer_idx] = terminal
        self._buffer_idx += 1

    def reset(self):
        self._buffer_idx = 0

    @classmethod
    def get_buffer(cls, num_envs: int, horizon: int, policy: keras.Model) -> "Buffer":
        size = horizon
        if len(policy.inputs) > 1:
            obs = tuple(
                [obs_from_shape(num_envs, size, input_layer) for input_layer in policy.inputs]
            )
        else:
            obs = obs_from_shape(size, num_envs,  policy.inputs[0])
        action = np.zeros((size, num_envs, 1), dtype=np.int32)
        reward = np.zeros((size, num_envs, 1), dtype=np.float32)
        estimated_value = np.zeros((size, num_envs, 1), dtype=np.float32)
        next_estimated_value = np.zeros((size, num_envs, 1), dtype=np.float32)
        terminal = np.zeros((size, num_envs, 1), dtype=bool)
        return cls(obs, action, reward, estimated_value, next_estimated_value, terminal)


@dataclass
class PPOStats:
    total_steps: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    final_reward: list[float] = field(default_factory=list)
    num_of_episodes: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    loss_clip: list[float] = field(default_factory=list)
    loss_critic: list[float] = field(default_factory=list)
    loss_entropy: list[float] = field(default_factory=list)

    approx_kls: list[float] = field(default_factory=list)
    clip_fractions: list[float] = field(default_factory=list)

    
    grad_norm: list[float] = field(default_factory=list)

    advs: list[float] = field(default_factory=list)

    def update(self, total_steps: int, buffer: Buffer, losses, loss_clip, loss_critic, loss_entropy, aprox_kl, clip_fraction, grad_norm, **kwargs):
        self.total_steps.append(total_steps)
        self.reward.append(np.mean(buffer.reward))
        self.final_reward.append(np.mean(buffer.reward[buffer.terminal]) if np.any(buffer.terminal) else 0)
        self.num_of_episodes.append(np.sum(buffer.terminal))
        self.value.append(np.mean(buffer.estimated_value))
        self.loss.append(np.mean(losses))

        self.loss_clip.append(np.mean(loss_clip))
        self.loss_critic.append(np.mean(loss_critic))
        self.loss_entropy.append(np.mean(loss_entropy))
        self.approx_kls.append(np.mean(aprox_kl))
        self.clip_fractions.append(np.mean(clip_fraction))

        self.grad_norm.append(np.mean(grad_norm))

        for field_name, values in kwargs.items():
            getattr(self, field_name).append(np.mean(values))


class AnnealingScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, final_learning_rate, annealing_steps):
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.annealing_steps = annealing_steps
        self.d_lr = (self.final_learning_rate - self.initial_learning_rate) / self.annealing_steps

    def __call__(self, step):
        if step < self.annealing_steps:
            return self.initial_learning_rate + self.d_lr * step
        return self.final_learning_rate


def obs_from_shape(num_envs: int, size: int, input_layer: keras.KerasTensor):
    """shape: (B, ...), model individual input shape, B is ignored"""
    dtype = TF_TO_NP_DTYPE[getattr(tf, input_layer.dtype)]
    return np.zeros((size, num_envs, *input_layer.shape[1:]), dtype=dtype)


def flat_envs_array(x: tuple | np.ndarray):
    if isinstance(x, tuple):
        return tuple([flat_envs_array(v) for v in x])
    return x.reshape((-1, *x.shape[2:]))


def feed_summary_writer(stats: PPOStats, start_time: int):
    total_steps = stats.total_steps[-1]
    tf.summary.scalar('charts/SPS', data=int(total_steps / (time.time() - start_time)), step=total_steps)

    tf.summary.scalar('losses/loss', data=stats.loss[-1], step=total_steps)
    tf.summary.scalar('losses/loss_clip', data=stats.loss_clip[-1], step=total_steps)
    tf.summary.scalar('losses/loss_critic', data=stats.loss_critic[-1], step=total_steps)
    tf.summary.scalar('losses/loss_entropy', data=stats.loss_entropy[-1], step=total_steps)

    tf.summary.scalar('losses/value', data=stats.value[-1], step=total_steps)
    tf.summary.scalar('losses/advs', data=stats.advs[-1], step=total_steps)

    tf.summary.scalar('losses/approx_kls', data=stats.approx_kls[-1], step=total_steps)
    tf.summary.scalar('losses/clip_fractions', data=stats.clip_fractions[-1], step=total_steps)

    tf.summary.scalar('env/reward', data=stats.reward[-1], step=total_steps)
    tf.summary.scalar('env/final_reward', data=stats.final_reward[-1], step=total_steps)
    tf.summary.scalar('env/num_of_episodes', data=stats.num_of_episodes[-1], step=total_steps)


def train(
        policy_fn: Callable[[], keras.Model], config: PPOHparams, envs: gym.vector.VectorEnv, logdir=None, env_seed=None
        ) -> tuple[keras.Model, PPOStats]: 
    start_time = time.time()

    if logdir is not None:
        writer = tf.summary.create_file_writer(logdir )
        writer.set_as_default()

    policy = policy_fn()
    old_policy = policy_fn()
    
    gae_estimator = vectorize_gae_estimator(get_gae_estimator(config.gamma, config.gae_lambda))
    
    loss_fn = build_loss(policy, old_policy, config, True)
    
    update_old_policy = old_policy_updater(policy, old_policy)
    buffer = Buffer.get_buffer(envs.num_envs, config.horizon, policy)
    num_iterations = config.total_steps // (config.horizon * envs.num_envs)
    obs, info = envs.reset(seed=env_seed)
    
    update_old_policy()
    
    if config.annealing_steps is not None:
        ann_scheduler = AnnealingScheduler(config.learning_rate, config.final_learning_rate, config.annealing_steps)
    else:
        ann_scheduler = None
    
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    stats = PPOStats()
    total_steps = 0
    
    old_obs = None

    for _ in tqdm(range(num_iterations)):
        while len(buffer) < config.horizon:
            # Would be nicer a for loop, but in the first iteration old_obs is None so the 
            ac_logits, next_value = policy(obs)
            if old_obs is not None:
                buffer.add(old_obs, action, tf.expand_dims(reward, axis=-1), 
                           e_value, next_value, tf.expand_dims(done, axis=-1))
            old_obs = obs
            e_value = next_value
            action = sample_logits(ac_logits)
    
            # step (transition) through the environment with the action
            # receiving the next observation, reward and if the episode has terminated or truncated
            obs, reward, terminated, truncated, info = envs.step(tf.squeeze(action))
            # If the episode has ended then we can reset to start a new episode
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                obs, info = envs.reset(options={"reset_mask": done})
            total_steps += envs.num_envs
    
        advantage = gae_estimator(buffer.estimated_value, buffer.next_estimated_value, buffer.reward, buffer.terminal)
        v_target = buffer.estimated_value + advantage
    
        advantage = flat_envs_array(advantage)
        v_target = flat_envs_array(v_target)
        data_obs = flat_envs_array(buffer.obs)
        data_actions = flat_envs_array(buffer.action)
    
        losses = []
        
        clip_losses = []
        critic_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        grad_norms = []

        advantages = []

        if ann_scheduler is not None:
            optimizer.learning_rate.assign(ann_scheduler(total_steps))
        for _ in range(config.epochs):
            shuffle_idx = np.random.permutation(len(v_target))
            batches_idx = np.split(shuffle_idx, len(v_target) // config.batch_size)
            for batch_idx in batches_idx:
                # Normalize the advantages
                if config.normalize_batch_advantage:
                    advs = (advantage[batch_idx] - advantage[batch_idx].mean()) / (advantage[batch_idx].std() + 1e-8)
                else:
                    advs = advantage[batch_idx]
                with tf.GradientTape() as tape:
                    loss, loss_clip, loss_critic, loss_entropy, aprox_kl, clip_fraction = loss_fn(
                        extract_obs_batch(batch_idx, data_obs),
                        data_actions[batch_idx],
                        advs,
                        v_target[batch_idx]
                    )
                    if tf.math.reduce_any(tf.math.is_nan(loss)):
                        raise RuntimeError("NANs found in loss!")
                grads = tape.gradient(loss, policy.trainable_variables)

                if config.clip_by_norm is not None:
                    grads, grads_global_norm = tf.clip_by_global_norm(grads, config.clip_by_norm)
                    # grads_global_norm does not consider potential clipping
                    grads_global_norm = tf.clip_by_value(grads_global_norm, 0, config.clip_by_norm)
                else:
                    grads_global_norm = tf.norm(tf.concat([tf.reshape(g, -1) for g in grads], 0))

                if tf.math.reduce_any(tf.math.is_nan(grads_global_norm)):
                    raise RuntimeError("NANs found in gradients!")
                optimizer.apply(grads, policy.trainable_variables)

                losses.append(loss.numpy())
                clip_losses.append(loss_clip.numpy())
                critic_losses.append(loss_critic.numpy())
                entropy_losses.append(loss_entropy.numpy())
                approx_kls.append(aprox_kl.numpy())
                clip_fractions.append(clip_fraction.numpy())
                grad_norms.append(grads_global_norm.numpy())
                advantages.append(np.std(advs))
        stats.update(total_steps, buffer, losses, clip_losses, critic_losses, entropy_losses, approx_kls, clip_fractions, grad_norms,
                    advs=advantages)
        if logdir is not None:
            feed_summary_writer(stats, start_time)

        buffer.reset()
        update_old_policy()
    return policy, stats

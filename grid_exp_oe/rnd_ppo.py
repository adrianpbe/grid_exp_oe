from collections.abc import Callable, Iterable, Sequence
import csv
from dataclasses import asdict, dataclass, field
import os
import time
from typing import Literal, overload

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm

from grid_exp_oe.base import AlgorithmHParams, AlgorithmRequirements
from grid_exp_oe.models import RNDActorCriticBuilder, PolicyType, prepare_rnn_inputs
from grid_exp_oe.common import sample_logits, get_gae_estimator, vectorize_gae_estimator
from grid_exp_oe.running_stats import RunningStats
import grid_exp_oe.ppo as ppo


RNDPPORequirements = AlgorithmRequirements(
    policy_type=PolicyType.RND_ACTOR_CRITIC
)


@dataclass
class RNDPPOHParams(ppo.PPOHparams):
    ext_coef: float = 1.0
    intrinsic_coef: float = 0.5
    intrinsic_gamma: float = 0.999
    intrinsic_episodic: bool = False
    proportion_distillation: float = 0.25
    initial_random_steps: int = 512
    substract_int_mean: bool = False

    def requirements(self) -> AlgorithmRequirements:
        return RNDPPORequirements

    def algo_id(self) -> str:
        return "rnd_ppo"


def build_rnd_loss(config: RNDPPOHParams):
    critic_coef, entropy_coef = config.critic_coef, config.entropy_coef
    policy_losses = ppo.build_ppo_policy_losses(config)

    def rnd_loss(logits, estimated_value, estimated_intrinsic_value, old_logits, actions, 
                 advantage, vtarget, intrinsic_vtarget, distillation_errors):
        loss_clip, loss_entropy, aprox_kl, clip_fraction = policy_losses(logits, old_logits, actions, advantage)

        loss_extrinsic_critic = tf.math.reduce_mean(tf.math.square(vtarget - estimated_value))
        loss_intrinsic_critic = tf.math.reduce_mean(tf.math.square(intrinsic_vtarget - estimated_intrinsic_value))

        loss_critic = 0.5 * (loss_extrinsic_critic + loss_intrinsic_critic)  # Multiplied to 0.5 to average the two losses

        mask = tf.random.uniform(tf.shape(distillation_errors)) < config.proportion_distillation
        mask = tf.cast(mask, tf.float32)

        distillation_loss = tf.math.reduce_sum(distillation_errors * mask) / tf.math.reduce_sum(mask)

        loss = loss_clip + critic_coef * loss_critic + entropy_coef * loss_entropy + distillation_loss

        return loss, loss_clip, loss_critic, loss_entropy, aprox_kl, clip_fraction, loss_extrinsic_critic, loss_intrinsic_critic, distillation_loss

    return tf.function(rnd_loss)


def build_rnd_loss_computation(policy: keras.Model, old_policy: keras.Model, rnd: keras.Model, config: RNDPPOHParams):
    loss_fn = build_rnd_loss(config)

    def loss_computation(batch):
        obs, actions, advantage, vtarget, intrinsic_vtarget, (rnd_obs_mean, rnd_obs_std) = batch
        logits, estimated_value, estimated_intrinsic_value = policy(obs)
        old_logits, *_ = old_policy(obs)
        old_logits = tf.stop_gradient(old_logits)
        distillation_errors = rnd((*obs, rnd_obs_mean, rnd_obs_std))
        return loss_fn(
            logits, estimated_value, estimated_intrinsic_value, old_logits,
            actions, advantage, vtarget, intrinsic_vtarget, distillation_errors)

    return loss_computation


@dataclass
class Buffer:
    obs: ppo.Obs
    next_obs: ppo.Obs
    action: np.ndarray
    reward: np.ndarray
    intrinsic_reward: np.ndarray
    estimated_value: np.ndarray
    estimated_intrinsic_value: np.ndarray
    next_estimated_value: np.ndarray
    next_estimated_intrinsic_value: np.ndarray
    terminal: np.ndarray
    old_terminal: np.ndarray  # used on RNN
    previous_states: None | tf.Tensor | list[tf.Tensor] = None

    def __len__(self):
        return self._buffer_idx

    def __post_init__(self):
        self._buffer_idx = 0
        self._max_length = len(self.terminal)
        if isinstance(self.obs, tuple):
            self._feed_obs = self._feed_multi_obs
        else:
            self._feed_obs = self._feed_single_obs

    def _feed_multi_obs(self, obs_field, obs: tuple[np.ndarray]):
        for buffer_ob, ob in zip(obs_field, obs):
            buffer_ob[self._buffer_idx] = ob

    def _feed_single_obs(self, obs_field, obs: np.ndarray):
        obs_field[self._buffer_idx] = obs

    def add(self, obs, next_obs, action, reward, value, intrinsic_value,
            next_value, next_intrinsic_value, terminal, old_terminal):
        if self._buffer_idx >=  self._max_length:
            raise RuntimeError("Buffer is full, it must be emptied before adding new data")
        self._feed_obs(self.obs, obs)
        self._feed_obs(self.next_obs, next_obs)
        self.action[self._buffer_idx] = action
        self.reward[self._buffer_idx] = reward
        self.estimated_value[self._buffer_idx] = value
        self.estimated_intrinsic_value[self._buffer_idx] = intrinsic_value
        self.next_estimated_value[self._buffer_idx] = next_value
        self.next_estimated_intrinsic_value[self._buffer_idx] = next_intrinsic_value
        self.terminal[self._buffer_idx] = terminal
        self.old_terminal[self._buffer_idx] = old_terminal
        self._buffer_idx += 1

    def reset(self):
        self._buffer_idx = 0

    @classmethod
    def get_buffer(cls, num_envs: int, horizon: int, policy: keras.Model) -> "Buffer":
        size = horizon
        obs = ppo.get_obs_from_policy(num_envs, policy.input, size)
        next_obs = ppo.get_obs_from_policy(num_envs, policy.input, size)
        action = np.zeros((size, num_envs), dtype=np.int32)
        reward = np.zeros((size, num_envs, 1), dtype=np.float32)
        intrinsic_reward = np.zeros((size, num_envs, 1), dtype=np.float32)
        estimated_value = np.zeros((size, num_envs, 1), dtype=np.float32)
        estimated_intrinsic_value = np.zeros((size, num_envs, 1), dtype=np.float32)
        next_estimated_value = np.zeros((size, num_envs, 1), dtype=np.float32)
        next_estimated_intrinsic_value = np.zeros((size, num_envs, 1), dtype=np.float32)
        terminal = np.zeros((size, num_envs, 1), dtype=bool)
        old_terminal = np.zeros((size, num_envs, 1), dtype=bool)
        return cls(obs, next_obs, action, reward, intrinsic_reward, estimated_value, estimated_intrinsic_value,
                   next_estimated_value, next_estimated_intrinsic_value, terminal, old_terminal)


@dataclass
class RNDStats:
    iteration: list[int] = field(default_factory=list)
    total_steps: list[int] = field(default_factory=list)
    steps_per_second: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    final_reward: list[float] = field(default_factory=list)
    intrinsic_reward: list[float] = field(default_factory=list)
    num_of_episodes: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    loss_clip: list[float] = field(default_factory=list)
    loss_critic: list[float] = field(default_factory=list)
    loss_extrinsic_critic: list[float] = field(default_factory=list)
    loss_intrinsic_critic: list[float] = field(default_factory=list)
    loss_distillation: list[float] = field(default_factory=list)

    loss_entropy: list[float] = field(default_factory=list)

    approx_kls: list[float] = field(default_factory=list)
    clip_fractions: list[float] = field(default_factory=list)

    grad_norm: list[float] = field(default_factory=list)

    advs: list[float] = field(default_factory=list)

    intrinsic_ret_mean: list[float] = field(default_factory=list)
    intrinsic_ret_std: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.start_time: float = time.time()

    def update(self, n_iteration: int, total_steps: int, buffer: Buffer, losses, loss_clip, loss_critic, loss_entropy, aprox_kl, clip_fraction, grad_norm, **kwargs):
        self.iteration.append(n_iteration)
        self.total_steps.append(total_steps)
        self.steps_per_second.append(total_steps / (time.time() - self.start_time))
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

    def last_iteration_stats(self) -> dict[str, float]:
        return {
            "iteration": self.iteration[-1],
            "env/total_steps": self.total_steps[-1],
            "env/steps_per_second": self.steps_per_second[-1],
            "env/reward": self.reward[-1],
            "env/final_reward": self.final_reward[-1],
            "env/intrinsic_reward": self.intrinsic_reward[-1],
            "env/num_of_episodes": self.num_of_episodes[-1],
            "losses/loss": self.loss[-1],
            "losses/loss_clip": self.loss_clip[-1],
            "losses/loss_critic": self.loss_critic[-1],
            "losses/loss_entropy": self.loss_entropy[-1],
            "losses/value": self.value[-1],
            "losses/advs": self.advs[-1],
            "losses/approx_kls": self.approx_kls[-1],
            "losses/clip_fractions": self.clip_fractions[-1],
            "losses/grad_norm": self.grad_norm[-1],
            "losses/loss_extrinsic_critic": self.loss_extrinsic_critic[-1],
            "losses/loss_intrinsic_critic": self.loss_intrinsic_critic[-1],
            "losses/loss_distillation": self.loss_distillation[-1],
            "losses/intrinsic_ret_mean": self.intrinsic_ret_mean[-1],
            "losses/intrinsic_ret_std": self.intrinsic_ret_std[-1],
            
        }

    @staticmethod
    def fieldnames() -> list[str]:
        return [
            "iteration",
            "env/total_steps",
            "env/steps_per_second",
            "env/reward",
            "env/final_reward",
            "env/intrinsic_reward",
            "env/num_of_episodes",
            "losses/loss",
            "losses/loss_clip",
            "losses/loss_critic",
            "losses/loss_entropy",
            "losses/value",
            "losses/advs",
            "losses/approx_kls",
            "losses/clip_fractions",
            "losses/grad_norm",
            "losses/loss_extrinsic_critic",
            "losses/loss_intrinsic_critic",
            "losses/loss_distillation",
            "losses/intrinsic_ret_mean",
            "losses/intrinsic_ret_std"
        ]


FIELDNAMES = RNDStats.fieldnames()


def vectorized_returns(rewards: np.ndarray, terminal, gamma: float, last_returns=None):
    returns = np.zeros_like(rewards)
    for i in range(len(rewards[1]) - 1, -1, -1):
        if last_returns is None:
            returns[i] = rewards[i]
        else:
            returns[i] = rewards[i] + gamma * (1 - terminal[i]) * returns[i + 1]
    return returns


def get_batches_iterator_fn(
        config: RNDPPOHParams, buffer: Buffer, gae_estimator, intrinsic_gae_estimator,
        intrinsic_running_stats: RunningStats,
        obs_running_stats: RunningStats,
        ) -> Callable[[], Iterable[ppo.TfBatches]]:
    extrinsic_advantage = gae_estimator(buffer.estimated_value, buffer.next_estimated_value, buffer.reward, buffer.terminal)
    # Normalize intrinsic reward
    if intrinsic_running_stats.shape != (1,):
        raise ValueError("Intrinsic running stats must be scalar")
    intrinsic_returns = vectorized_returns(buffer.intrinsic_reward, buffer.terminal, config.intrinsic_gamma)
    intrinsic_running_stats.batch_update(intrinsic_returns.reshape(-1))
    if config.substract_int_mean:
        intrinsic_reward = intrinsic_running_stats.normalize(buffer.intrinsic_reward)
    else:
        # According to the paper, intrinsic rewards are normalized by dividing by the standard deviation, mean is ignored
        intrinsic_reward = buffer.intrinsic_reward / (intrinsic_running_stats.std + 1e-8)

    if config.intrinsic_episodic:
        intrinsic_advantage = intrinsic_gae_estimator(
            buffer.estimated_intrinsic_value, buffer.next_estimated_intrinsic_value, intrinsic_reward, buffer.terminal
        )
    else:
        intrinsic_advantage = intrinsic_gae_estimator(
            buffer.estimated_intrinsic_value, buffer.next_estimated_intrinsic_value, intrinsic_reward, np.zeros_like(buffer.terminal)
        )

    v_target = buffer.estimated_value + extrinsic_advantage
    v_target_intrinsic = buffer.estimated_intrinsic_value + intrinsic_advantage
    advantage = extrinsic_advantage * config.ext_coef + intrinsic_advantage * config.intrinsic_coef
    advantage = ppo.flat_envs_array(advantage)
    v_target = ppo.flat_envs_array(v_target)
    v_target_intrinsic = ppo.flat_envs_array(v_target_intrinsic)
    data_obs = ppo.flat_envs_array(buffer.obs)
    data_actions = ppo.flat_envs_array(buffer.action)
    indexes = tf.range(tf.shape(advantage)[0])

    def get_batches() -> Iterable[ppo.TfBatches]:
        shuffle_idx = tf.random.shuffle(indexes)
        batches_idx = tf.split(shuffle_idx, config.num_batches)
        for batch_idx in batches_idx:
            advs = tf.gather(advantage, batch_idx)
            if config.normalize_batch_advantage:
                advs = (advs - tf.math.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
            yield (
                ppo.tf_extract_obs_batch(batch_idx, data_obs), tf.gather(data_actions, batch_idx), advs, 
                tf.gather(v_target, batch_idx), tf.gather(v_target_intrinsic, batch_idx), 
                (tf.convert_to_tensor(obs_running_stats.mean), tf.convert_to_tensor(obs_running_stats.std))
            )
    
    return get_batches


def collects_random_stats(env: gym.vector.VectorEnv, num_steps: int, runnin_stats: RunningStats):
    for _ in range(num_steps):
        acs = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(acs)

        # If the episode has ended then we can reset to start a new episode
        done = np.logical_or(terminated, truncated)
        if np.any(done):
            obs, info = env.reset(options={"reset_mask": done})
        runnin_stats.batch_update(obs[0].reshape(-1))


def rnd_ppo_train(
        model_builder: RNDActorCriticBuilder, config: RNDPPOHParams, envs: gym.vector.VectorEnv,
        save_freq=1, *, experimentdir=None,
        ckptdir=None, logdir=None, env_seed=None
        ) -> tuple[keras.Model, RNDStats, dict]:
    profiling = ppo.PPOProfiling()
    policy_type = model_builder.policy_type()

    if policy_type != RNDPPORequirements.policy_type:
        raise ValueError(f"policy type {policy_type} not supported in RND PPO")

    if experimentdir is not None:
        statsdir = os.path.join(experimentdir, "stats.csv")
        feed_stats_csv = ppo.get_feed_stats_csv(statsdir, FIELDNAMES)
        with open(statsdir, "w", newline='') as f:
            csv_writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            csv_writer.writeheader()

    if logdir is not None:
        tensor_writer = tf.summary.create_file_writer(logdir)
        tensor_writer.set_as_default()

    policy = model_builder.build(envs)
    old_policy = model_builder.build(envs)

    rnd = model_builder.rnd()

    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    # Despite the obs being an image, the running stats are scalar, so
    #  images are normalized by the mean and std of all pixels and channels.
    # Other observations componentes (as directions) are not normalized
    obs_running_stats = RunningStats(shape=(1,))
    int_reward_running_stats = RunningStats(shape=(1,))

    if ckptdir is not None:
        checkpoint_managers = ppo.create_checkpoints(ckptdir, policy, optimizer, save_freq, "rnd_ppo_policy", {"rnd_ppo": rnd})
    else:
        checkpoint_managers = None

    gae_estimator = vectorize_gae_estimator(get_gae_estimator(config.gamma, config.gae_lambda))
    intrinsic_gae_estimator = vectorize_gae_estimator(get_gae_estimator(config.intrinsic_gamma, config.gae_lambda))

    loss_fn = build_rnd_loss_computation(policy, old_policy, rnd, config)
    get_batches = get_batches_iterator_fn
    policy_fn = lambda obs_, states_, starts_: (*policy(obs_), None)
    states = None

    update_old_policy = ppo.old_policy_updater(policy, old_policy)
    
    buffer = Buffer.get_buffer(envs.num_envs, config.horizon, policy)

    num_iterations = config.total_steps // (config.horizon * envs.num_envs)
    obs, info = envs.reset(seed=env_seed)
    
    update_old_policy()
    
    if config.annealing_steps is not None:
        ann_scheduler = ppo.AnnealingScheduler(config.learning_rate, config.final_learning_rate, config.annealing_steps)
    else:
        ann_scheduler = None
    
    stats = RNDStats()

    if config.initial_random_steps > 0:
        collects_random_stats(envs, config.initial_random_steps, obs_running_stats)
        obs, info = envs.reset()

    total_steps = 0

    old_obs = None
    old_done = None
    done = np.ones(config.num_envs, dtype=bool)

    for n_iteration in tqdm(range(num_iterations)):
        with profiling.profile_rollout():
            buffer.previous_states = states
            while len(buffer) < config.horizon:
                # Would be nicer a for loop, but in the first iteration old_obs is None, so no
                #  data is added to the buffer and an additional loop step is required to fill it up
                done = tf.expand_dims(done, axis=-1)
                ac_logits, next_value, next_intrinsic_value, states = policy_fn(obs, states, done)
                if old_obs is not None:
                    buffer.add(old_obs, obs, action, tf.expand_dims(reward, axis=-1), 
                            e_value, e_intrinsic_value, next_value, next_intrinsic_value, done,
                            old_done
                            )
                old_obs = obs
                e_value = next_value
                e_intrinsic_value = next_intrinsic_value
                old_done = done
                action = sample_logits(ac_logits)

                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                obs, reward, terminated, truncated, info = envs.step(action)
                # If the episode has ended then we can reset to start a new episode
                done = np.logical_or(terminated, truncated)
                if np.any(done):
                    obs, info = envs.reset(options={"reset_mask": done})
                
                obs_running_stats.batch_update(obs[0].reshape(-1))
                total_steps += envs.num_envs

            # Rollouts intrinsic reward, it's computed afterwards to vectorize the computation
            obrnd = (*ppo.flat_envs_array(buffer.obs, False), obs_running_stats.mean, obs_running_stats.std)
            int_reward = rnd(obrnd)
            buffer.intrinsic_reward = tf.reshape(int_reward, [config.horizon, envs.num_envs, 1]).numpy()

        with profiling.profile_training():
            iterator_fn = get_batches(config, buffer, gae_estimator, intrinsic_gae_estimator, int_reward_running_stats, obs_running_stats)
            losses = []
            clip_losses = []
            critic_losses = []
            entropy_losses = []
            approx_kls = []
            clip_fractions = []
            extrinsic_critic_losses = []
            intrinsic_critic_losses = []
            distillation_losses = []
            grad_norms = []
            advantages = []

            if ann_scheduler is not None:
                optimizer.learning_rate.assign(ann_scheduler(total_steps))
            for _ in range(config.epochs):
                batches = iterator_fn()
                for batch in batches:
                    with tf.GradientTape() as tape:
                        (
                            loss, loss_clip, loss_critic, loss_entropy, aprox_kl, clip_fraction,
                            loss_extrinsic_critic, loss_intrinsic_critic, distillation_loss
                         ) = loss_fn(batch)
                        if tf.math.reduce_any(tf.math.is_nan(loss)):
                            raise RuntimeError("NANs found in loss!")
                    grads = tape.gradient(loss, policy.trainable_variables + rnd.trainable_variables)

                    if config.clip_by_norm is not None:
                        grads, grads_global_norm = tf.clip_by_global_norm(grads, config.clip_by_norm)
                        # grads_global_norm does not consider potential clipping
                        grads_global_norm = tf.clip_by_value(grads_global_norm, 0, config.clip_by_norm)
                    else:
                        grads_global_norm = tf.norm(tf.concat([tf.reshape(g, -1) for g in grads], 0))

                    if tf.math.reduce_any(tf.math.is_nan(grads_global_norm)):
                        raise RuntimeError("NANs found in gradients!")
                    optimizer.apply(grads, policy.trainable_variables + rnd.trainable_variables)

                    batch_adv = batch[3]

                    losses.append(loss.numpy())
                    clip_losses.append(loss_clip.numpy())
                    critic_losses.append(loss_critic.numpy())
                    entropy_losses.append(loss_entropy.numpy())
                    approx_kls.append(aprox_kl.numpy())
                    clip_fractions.append(clip_fraction.numpy())
                    extrinsic_critic_losses.append(loss_extrinsic_critic.numpy())
                    intrinsic_critic_losses.append(loss_intrinsic_critic.numpy())
                    distillation_losses.append(distillation_loss.numpy())
                    grad_norms.append(grads_global_norm.numpy())
                    advantages.append(np.std(batch_adv))

            stats.update(
                n_iteration, total_steps, buffer, losses,
                clip_losses, critic_losses, entropy_losses, 
                approx_kls, clip_fractions, grad_norms,
                advs=advantages,
                loss_extrinsic_critic=extrinsic_critic_losses,
                loss_intrinsic_critic=intrinsic_critic_losses,
                loss_distillation=distillation_losses,
                intrinsic_reward=buffer.intrinsic_reward,
                intrinsic_ret_mean=int_reward_running_stats.mean,
                intrinsic_ret_std=int_reward_running_stats.std,
                )

        iter_stats = stats.last_iteration_stats()
        if experimentdir is not None:
            feed_stats_csv(iter_stats)
        if logdir is not None:
            ppo.feed_summary_writer(iter_stats)
        if ckptdir is not None:
            ppo.update_checkpoint(checkpoint_managers, n_iteration, buffer.reward.mean())
        buffer.reset()
        update_old_policy()
    profiling.close()
    return policy, stats, {"profiling": asdict(profiling)}

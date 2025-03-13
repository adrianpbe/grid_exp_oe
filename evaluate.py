from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from simple_parsing import ArgumentParser
import tensorflow as tf
from tqdm import tqdm

from grid_exp_oe.env import create_vectorized_env, create_env
from grid_exp_oe.ppo import sample_logits
from train import get_features_extractor, build_policy


SUPPORTED_ALGORITHMS = {
    "random",
    "ppo",
}


@dataclass
class EvaluationConfig:
    env_id: str
    algorithm: str = "random"
    ckpt: str | None = None
    total_steps: int = 10000
    num_envs: int = 1
    env_seed: int | None = None
    human_render: bool = False

    def __post_init__(self):
        if self.human_render and self.num_envs > 1:
            self.num_envs = 1


def get_policy(config: EvaluationConfig, env: gym.Env, expand_batch: bool):
    if config.algorithm == "ppo":
        if config.ckpt is None:
            raise ValueError("ckpt must be provided")
        actor_critic_model = build_policy(get_features_extractor())
        checkpoint = tf.train.Checkpoint(model=actor_critic_model)
        checkpoint.restore(config.ckpt).expect_partial()  # expect_partial() suppresses warnings about not restoring all variables
        print(f"restored ckpts: {config.ckpt}")
        def policy(obs):
            if expand_batch:
                obs = (tf.expand_dims(obs[0], axis=0), tf.expand_dims(obs[1], axis=0))
            logits, _ = actor_critic_model(obs)
            return tf.squeeze(sample_logits(logits)).numpy()
    elif config.algorithm == "random":
        policy = lambda _obs: env.action_space.sample()
    return policy


def evaluate(config: EvaluationConfig):
    if config.algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Not supported {config.algorithm} algorithm")
    total_steps = config.total_steps
    num_envs = config.num_envs
    if num_envs > 1:
        rewards, dones = vectorized_eval(config)
    else:
        rewards, dones = single_env_eval(config)


    print("Total steps: ", total_steps)
    print("Total episodes:", dones.sum())
    print(f"Mean reward: {rewards.mean():.3f}")
    print(f"Std reward: {rewards.std():.3f}")

    if dones.any():  # Avoid potential error if no episodes are done
        print(f"Mean final reward: {rewards[dones].mean():.3f}")
        print(f"Std final reward: {rewards[dones].std():.3f}")
    else:
        print("No completed episodes to calculate final rewards.")


def vectorized_eval(config: EvaluationConfig):
    envs = create_vectorized_env(config.env_id, config.num_envs)

    d = deque()

    policy = get_policy(config, envs, False)

    obs, info = envs.reset()

    num_iterations = config.total_steps //  config.num_envs
    has_started = False

    for iteration in tqdm(range(num_iterations)):
        action = policy(obs)
        obs, reward, terminated, truncated, info = envs.step(action)
        if has_started:
            d.append(
                (
                    reward,
                    np.logical_or(terminated, truncated)
                )
            )
        else:
            has_started = True

        # If the episode has ended then we can reset to start a new episode
        done = np.logical_or(terminated, truncated)
        if np.any(done):
            obs, info = envs.reset(options={"reset_mask": done})
    
    rewards, dones = zip(*list(d))
    rewards = np.stack(rewards)
    dones = np.stack(dones)
    return rewards, dones


def single_env_eval(config: EvaluationConfig):
    env = create_env(config.env_id, 
                     render_mode="human" if config.human_render else None)
    total_steps = config.total_steps

    policy = get_policy(config, env, expand_batch=True)

    d = deque()
    obs, _ = env.reset()
    for iteration in tqdm(range(total_steps)):        
        action = policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        d.append(
            (
                reward,
                terminated or truncated
            )
        )
        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            obs, info = env.reset()

    rewards, dones = zip(*list(d))
    rewards = np.stack(rewards)
    dones = np.stack(dones)
    return rewards,dones


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(EvaluationConfig, dest="config")
    args = parser.parse_args()
    
    config: EvaluationConfig = args.config

    evaluate(config)

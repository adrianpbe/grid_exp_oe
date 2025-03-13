from collections import deque
from dataclasses import dataclass
import numpy as np
from simple_parsing import ArgumentParser
from tqdm import tqdm

from grid_exp_oe.env import create_vectorized_env, create_env


@dataclass
class EvaluationConfig:
    env_id: str
    algorithm: str = "random"
    total_steps: int = 10000
    num_envs: int = 1
    env_seed: int | None = None
    human_render: bool = False

    def __post_init__(self):
        if self.human_render and self.num_envs > 1:
            self.num_envs = 1


def evaluate(config: EvaluationConfig):
    if config.algorithm != "random":
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

    # Just a random policy
    policy = lambda _obs: envs.action_space.sample()

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
    # Just a random policy
    policy = lambda _obs: env.action_space.sample()

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

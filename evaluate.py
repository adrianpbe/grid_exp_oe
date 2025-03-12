from collections import deque
from dataclasses import dataclass

import numpy as np
from simple_parsing import ArgumentParser
from tqdm import tqdm

from grid_exp_oe.env import create_vectorize_env


@dataclass
class EvaluationConfig:
    env_id: str
    algorithm: str = "random"
    total_steps: int = 10000
    num_envs: int = 16
    env_seed: int | None = None


def evaluate(config: EvaluationConfig):
    if config.algorithm != "random":
        raise ValueError(f"Not supported {config.algorithm} algorithm")
    total_steps = config.total_steps
    num_envs = config.num_envs
    envs = create_vectorize_env(config.env_id, config.num_envs)

    d = deque()

    # Just a random policy
    policy = lambda _obs: envs.action_space.sample()

    obs, info = envs.reset()

    num_iterations = total_steps //  num_envs
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

    print("Total steps: ", total_steps)
    print("Total episodes:", dones.sum())
    print(f"Mean reward: {rewards.mean():.3f}")
    print(f"Std reward: {rewards.std():.3f}")

    if dones.any():  # Avoid potential error if no episodes are done
        print(f"Mean final reward: {rewards[dones].mean():.3f}")
        print(f"Std final reward: {rewards[dones].std():.3f}")
    else:
        print("No completed episodes to calculate final rewards.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(EvaluationConfig, dest="config")
    args = parser.parse_args()
    
    config: EvaluationConfig = args.config

    evaluate(config)

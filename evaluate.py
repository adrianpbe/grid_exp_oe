from collections import deque
from dataclasses import dataclass
import json
import os

import gymnasium as gym
import numpy as np
from simple_parsing import ArgumentParser
import tensorflow as tf
from tqdm import tqdm

from grid_exp_oe.env import create_vectorized_env, create_env
from grid_exp_oe.models import get_model_builder


@dataclass
class EvaluationConfig:
    experiment: str | None = None
    env_id: str | None = None
    random: bool = False
    ckpt: str | None = None
    total_steps: int = 10000
    num_envs: int = 1
    env_seed: int | None = None
    human_render: bool = False

    def __post_init__(self):
        if self.human_render and self.num_envs > 1:
            self.num_envs = 1


def get_policy(config: EvaluationConfig, env: gym.Env, expand_batch: bool, exp_data: dict | None):
    if config.experiment is not None:
        if config.random:
            print("random flag is activated but will be ignored as experiment is provided")
        if config.ckpt is None:
            print("no ckpt is provided, the best policy ckpt found will be used")
            ckpt_path = tf.train.latest_checkpoint(os.path.join(config.experiment, "best_checkpoints"))
            if ckpt_path is None:
                raise ValueError("not ckpt found")
        else:
            ckpt_path = config.ckpt
        model_id = exp_data["model"]["model_id"]
        model_hparams_dict = exp_data["model"]["hparams"]
        model_builder = get_model_builder(model_id, model_hparams_dict)
        model = model_builder.build(env)
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(ckpt_path).expect_partial()  # expect_partial() suppresses warnings about not restoring all variables
        print(f"restored ckpts: {ckpt_path}")
        policy = model_builder.adapt_eval(model, config.num_envs, expand_batch, return_numpy=True)
    else:
        policy = lambda _obs, start: env.action_space.sample()
    return policy


def evaluate(config: EvaluationConfig):
    if config.random and config.env_id is None:
        raise ValueError("if random flag is activated an env_id is required")
    if config.experiment is not None:
        with open(os.path.join(config.experiment, "config.json"), "r") as f:
            exp_data = json.load(f)
        if config.env_id is None:
            config.env_id = exp_data["experiment"]["env_id"]
    else:
        exp_data = None
    total_steps = config.total_steps
    num_envs = config.num_envs
    if num_envs > 1:
        rewards, dones = vectorized_eval(config, exp_data)
    else:
        rewards, dones = single_env_eval(config, exp_data)


    print("Total steps: ", total_steps)
    print("Total episodes:", dones.sum())
    print(f"Mean reward: {rewards.mean():.3f}")
    print(f"Std reward: {rewards.std():.3f}")

    if dones.any():  # Avoid potential error if no episodes are done
        print(f"Mean final reward: {rewards[dones].mean():.3f}")
        print(f"Std final reward: {rewards[dones].std():.3f}")
    else:
        print("No completed episodes to calculate final rewards.")


def vectorized_eval(config: EvaluationConfig, exp_data):
    envs = create_vectorized_env(config.env_id, config.num_envs)

    d = deque()

    policy = get_policy(config, envs, False, exp_data)

    obs, info = envs.reset()

    num_iterations = config.total_steps //  config.num_envs
    has_started = False
    # initiliazed dones because they are used in policy to indicate the beginning of
    #  espisodes
    done = np.ones(config.num_envs, dtype=bool)
    for iteration in tqdm(range(num_iterations)):
        action = policy(obs, done)
        obs, reward, terminated, truncated, info = envs.step(action)
        done = np.logical_or(terminated, truncated)
        if has_started:
            d.append(
                (
                    reward,
                    done
                )
            )
        else:
            has_started = True

        # If the episode has ended then we can reset to start a new episode
        if np.any(done):
            obs, info = envs.reset(options={"reset_mask": done})
    
    rewards, dones = zip(*list(d))
    rewards = np.stack(rewards)
    dones = np.stack(dones)
    return rewards, dones


def single_env_eval(config: EvaluationConfig, exp_data):
    env = create_env(config.env_id, 
                     render_mode="human" if config.human_render else None)
    total_steps = config.total_steps

    policy = get_policy(config, env, expand_batch=True, exp_data=exp_data)

    d = deque()
    obs, _ = env.reset()
    # initiliazed done because it is used in policy to indicate the beginning of
    #  espisodes
    done = True
    for iteration in tqdm(range(total_steps)):        
        action = policy(obs, done)
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        d.append(
            (
                reward,
                done,
            )
        )
        # If the episode has ended then we can reset to start a new episode
        if done:
            obs, info = env.reset()

    rewards, dones = zip(*list(d))
    rewards = np.stack(rewards)
    dones = np.stack(dones)
    return rewards, dones


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(EvaluationConfig, dest="config")
    args = parser.parse_args()
    
    config: EvaluationConfig = args.config

    evaluate(config)

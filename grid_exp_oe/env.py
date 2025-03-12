import gymnasium as gym
import minigrid
import numpy as np


def process_obs(obs):
    return obs["image"], np.array(obs["direction"])


def create_vectorize_env(env_id: str, num_envs: int):
    """Creates a vectorized Minigrid environment, whose observations are a tuple with the image and the direction.
    Nothing prevents you from creating a non Minigrid env, but the observation space of the environment with
    `env_id` must a Dict with at least the "image" and "direction" fields.
    """
    check_obs_env = gym.make(env_id)
    if not isinstance(check_obs_env.observation_space, gym.spaces.Dict):
        raise ValueError(f"env {env_id} observation_space must be a dict")
    for k in ["image", "direction"]:
        if k not in check_obs_env.observation_space.keys():
            raise ValueError(f"{k} must be an obeservation_space key")
    envs_fns = [lambda: gym.make(env_id) for _ in range(num_envs)]
    oenvs = gym.vector.SyncVectorEnv(envs_fns, autoreset_mode=gym.vector.AutoresetMode.DISABLED)

    envs = gym.wrappers.vector.VectorizeTransformObservation(
        oenvs,
        wrapper=gym.wrappers.TransformObservation,
        func=process_obs,
        observation_space=gym.spaces.Tuple(
                    (oenvs.single_observation_space["image"], oenvs.single_observation_space["direction"])
        ),
    )
    return envs


from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

from simple_parsing import ArgumentParser
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from grid_exp_oe.env import create_vectorize_env
from grid_exp_oe.ppo import PPOHparams, train


IM_OBS_SHAPE = (7, 7, 3)
NUM_ACTIONS = 7


def get_features_extractor():
    return keras.models.Sequential(
        [
            layers.Input(shape=IM_OBS_SHAPE, dtype=tf.float32),
            layers.Conv2D(8, 3, activation="relu", padding="same"),
            layers.Conv2D(16, 3, activation="relu", padding="same", strides=2),
            layers.Conv2D(16, 3, activation="relu", padding="same"),
            layers.Reshape((16, 16)),
        ]
    )


def build_policy(feature_extractor):
    in_img_obs = layers.Input(shape=(7,7,3), dtype=tf.float32)
    dir_input = layers.Input(shape=(), dtype=tf.int32)
    dir_encoder = layers.Embedding(4, 32)
    dir_z = dir_encoder(dir_input)
    
    x = feature_extractor(in_img_obs)
    
    x = layers.MultiHeadAttention(4, 16)(x, x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = x + dir_z
    
    z_policy = layers.Dense(32, activation="relu")(x)
    logits = layers.Dense(NUM_ACTIONS, activation=None)(z_policy)
    
    z_critic = layers.Dense(32, activation="relu")(x)
    value = layers.Dense(1, activation=None)(z_critic)
    
    policy = keras.Model(inputs=(in_img_obs, dir_input), outputs=(logits, value))
    return policy


@dataclass
class ExperimentConfig:
    env_id: str
    name: str | None = None
    env_seed: int | None = None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment")
    parser.add_arguments(PPOHparams, dest="hparams")
    parser.add_argument("--logdir", type=str, default="logs", 
                        help="path to Tensorboard logs folder (a new folder for the current experiment is created within)")
    parser.add_argument("--expdir", type=str, default="experiments", 
                        help="path to experiments folder (a new folder for the current experiment is created within)")
    args = parser.parse_args()
    
    experiment: ExperimentConfig = args.experiment
    hparams: PPOHparams = args.hparams

    exp_name = experiment.name
    exp_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    logdir = os.path.join(
        args.logdir,
        exp_name + "_" + exp_time_str if exp_name is not None else exp_time_str 
    )

    expdir = os.path.join(
        args.expdir,
        exp_name + "_" + exp_time_str if exp_name is not None else exp_time_str 
    )
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.expdir, exist_ok=True)

    os.makedirs(logdir)
    os.makedirs(expdir)

    all_configs = {
        "experiment": asdict(experiment),
        "hparams": asdict(hparams),
        "metadata": {
            "time": exp_time_str
        }
    }
    store_cfg = os.path.join(expdir, "config.json")
    with open(store_cfg, "w") as f:
        json.dump(all_configs, f, indent=4)

    envs = create_vectorize_env(experiment.env_id, hparams.num_envs)

    get_policy_fn = lambda: build_policy(get_features_extractor())

    policy, stats = train(
        get_policy_fn, hparams, envs,
        logdir=logdir,
        env_seed=experiment.env_seed
    )

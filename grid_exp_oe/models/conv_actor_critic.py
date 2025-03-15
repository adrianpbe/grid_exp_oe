from dataclasses import dataclass, field

import gymnasium as gym
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from grid_exp_oe.models.base import ActorCriticBuilder, ModelHparams


@dataclass
class ConvActorCriticHParams(ModelHparams):
    policy_units: int = 32
    critic_units: int = 32
    dense_layers: list[int] = field(default_factory=lambda: [32])
    direction_emb_size: int = 32
    attention: bool = True
    attention_units: int = 16
    activation: str = "relu"


def configure_in_out_shapes(env: gym.Env | gym.vector.VectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = env.observation_space if isinstance(env, gym.Env) else env.single_observation_space
    ac_space = env.action_space if isinstance(env, gym.Env) else env.single_action_space
    if not isinstance(obs_space, gym.spaces.Tuple) or len(obs_space) != 2:
        raise ValueError("this model only accepts tuple observation_spaces of two elements")
    if not isinstance(ac_space, gym.spaces.Discrete):
        raise ValueError("this model only accepts discrete action_spaces")

    im_obs, dir_obs = obs_space
    num_actions = int(ac_space.n)

    return im_obs.shape, num_actions


def get_features_extractor(im_obs_shape, hparams: ConvActorCriticHParams):
    return keras.models.Sequential(
        [
            layers.Input(shape=im_obs_shape, dtype=tf.float32),
            layers.Conv2D(8, 3, activation=hparams.activation, padding="same"),
            layers.Conv2D(16, 3, activation=hparams.activation, padding="same", strides=2),
            layers.Conv2D(16, 3, activation=hparams.activation, padding="same"),
        ]
    )


class ConvActorCriticBuilder(ActorCriticBuilder[ConvActorCriticHParams]):
    def __init__(self, hparams: ConvActorCriticHParams):
        self.hparams = hparams

        self.im_obs_shape = None
        self.num_actions = None

    def _build(self, env:  gym.Env | gym.vector.VectorEnv) -> keras.Model:
        self.im_obs_shape, self.num_actions = configure_in_out_shapes(env)
        hparams: ConvActorCriticHParams = self.hparams
        feature_extractor = get_features_extractor(self.im_obs_shape, hparams)
        in_img_obs = layers.Input(shape=self.im_obs_shape, dtype=tf.float32)
        dir_input = layers.Input(shape=(), dtype=tf.int32)
        dir_encoder = layers.Embedding(4, hparams.direction_emb_size)
        dir_z = dir_encoder(dir_input)

        x = feature_extractor(in_img_obs)

        if hparams.attention:
            # features_maps_shape = tf.shape(x)
            # features_channels = features_maps_shape[-1]
            # num_spatial_features = tf.math.reduce_prod(features_maps_shape[-3:-1])
            x = layers.Reshape((16, 16))(x)
            x = layers.MultiHeadAttention(4, hparams.attention_units)(x, x)
            x = layers.Dense(hparams.attention_units, activation=hparams.activation)(x)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.Flatten()(x)

        for units in hparams.dense_layers:
            x = layers.Dense(units, activation=hparams.activation)(x)

        x = x + dir_z

        z_policy = layers.Dense(hparams.policy_units, activation=hparams.activation)(x)
        logits = layers.Dense(self.num_actions, activation=None)(z_policy)

        z_critic = layers.Dense(hparams.critic_units, activation=hparams.activation)(x)
        value = layers.Dense(1, activation=None)(z_critic)

        return (in_img_obs, dir_input), (logits, value)

    @staticmethod
    def HParams() -> type[ConvActorCriticHParams]:
        return ConvActorCriticHParams

    @staticmethod
    def model_id():
        return "conv_actor_critic"

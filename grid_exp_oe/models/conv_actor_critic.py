from dataclasses import dataclass, field

import gymnasium as gym
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from grid_exp_oe.models.base import ActorCriticBuilder, ModelHparams, RNDActorCriticBuilder


@dataclass
class ConvActorCriticHParams(ModelHparams):
    policy_units: int = 32
    critic_units: int = 32
    dense_layers: list[int] = field(default_factory=lambda: [32])
    attention: bool = True
    attention_units: int = 16
    activation: str = "relu"



@dataclass
class RNDConvActorCriticHParams(ConvActorCriticHParams):
    rnd_emb_size: int = 32


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


def build_encoder(im_obs_shape, hparams):
    feature_extractor = get_features_extractor(im_obs_shape, hparams)
    in_img_obs = layers.Input(shape=im_obs_shape, dtype=tf.float32)
    dir_input = layers.Input(shape=(), dtype=tf.int32)

    dir_encoder = layers.Embedding(4, hparams.attention_units if hparams.attention else 16)
    dir_z = dir_encoder(dir_input)

    x = feature_extractor(in_img_obs)

    if hparams.attention:
        x = layers.Reshape((-1, 16))(x)
        x = layers.MultiHeadAttention(4, hparams.attention_units)(x, x)
        x = layers.Dense(hparams.attention_units, activation=hparams.activation)(x)
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = layers.Flatten()(x)

    x = x + dir_z

    for units in hparams.dense_layers:
        x = layers.Dense(units, activation=hparams.activation)(x)
    return (in_img_obs,dir_input), x


class ConvActorCriticBuilder(ActorCriticBuilder[ConvActorCriticHParams]):
    def __init__(self, hparams: ConvActorCriticHParams):
        self.hparams = hparams

        self.im_obs_shape = None
        self.num_actions = None

    def _build(self, env:  gym.Env | gym.vector.VectorEnv) -> keras.Model:
        self.im_obs_shape, self.num_actions = configure_in_out_shapes(env)
        im_obs_shape, num_actions =  self.im_obs_shape, self.num_actions
        hparams: ConvActorCriticHParams = self.hparams

        (in_img_obs, dir_input), x = build_encoder(im_obs_shape, hparams)

        z_policy = layers.Dense(hparams.policy_units, activation=hparams.activation)(x)
        logits = layers.Dense(num_actions, activation=None)(z_policy)

        z_critic = layers.Dense(hparams.critic_units, activation=hparams.activation)(x)
        value = layers.Dense(1, activation=None)(z_critic)

        return (in_img_obs, dir_input), (logits, value)


    @staticmethod
    def HParams() -> type[ConvActorCriticHParams]:
        return ConvActorCriticHParams

    @staticmethod
    def model_id():
        return "conv_actor_critic"


def build_rnd_forward_model(im_obs_shape, emb_size: int, activation: str, name: str):
    feature_extractor = keras.models.Sequential(
        [
            layers.Input(shape=im_obs_shape, dtype=tf.float32),
            layers.Conv2D(8, 3, activation=activation, padding="same"),
            layers.Conv2D(16, 3, activation=activation, padding="same", strides=2),
            layers.Conv2D(16, 3, activation=activation, padding="same"),
        ]
    )

    in_img_obs = layers.Input(shape=im_obs_shape, dtype=tf.float32)
    dir_input = layers.Input(shape=(), dtype=tf.int32)

    dir_encoder = layers.Embedding(4, 64)
    dir_z = dir_encoder(dir_input)
    x = feature_extractor(in_img_obs)

    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = x + dir_z

    x = layers.Dense(emb_size, activation=activation)(x)
    return keras.Model(inputs=(in_img_obs, dir_input), outputs=x, name=name)


def normalize_im(x):
    obs_im, rnd_obs_mean, rnd_obs_std = x
    norm_obs_im = (obs_im - rnd_obs_mean) / (rnd_obs_std + 1e-8)
    norm_obs_im = tf.clip_by_value(norm_obs_im, -5.0, 5.0)
    return norm_obs_im


class RNDConvActorCriticBuilder(RNDActorCriticBuilder[ConvActorCriticHParams]):
    def __init__(self, hparams: RNDConvActorCriticHParams):
        self.hparams = hparams

        self.im_obs_shape = None
        self.num_actions = None

    def _build(self, env:  gym.Env | gym.vector.VectorEnv) -> keras.Model:
        self.im_obs_shape, self.num_actions = configure_in_out_shapes(env)
        im_obs_shape, num_actions =  self.im_obs_shape, self.num_actions
        hparams: RNDConvActorCriticHParams = self.hparams

        (in_img_obs, dir_input), x = build_encoder(im_obs_shape, hparams)

        z_policy = layers.Dense(hparams.policy_units, activation=hparams.activation)(x)
        logits = layers.Dense(num_actions, activation=None)(z_policy)

        z_critic = layers.Dense(hparams.critic_units, activation=hparams.activation)(x)
        value = layers.Dense(1, activation=None)(z_critic)

        z_intrinic_critic = layers.Dense(hparams.critic_units, activation=hparams.activation)(x)
        intrinsic_value = layers.Dense(1, activation=None)(z_intrinic_critic)

        return (in_img_obs, dir_input), (logits, value, intrinsic_value)

    def rnd(self):
        in_img_obs = layers.Input(shape=self.im_obs_shape, dtype=tf.float32)
        dir_input = layers.Input(shape=(), dtype=tf.int32)

        im_mu = layers.Input(shape=(), batch_size=1, dtype=tf.float32)
        im_std = layers.Input(shape=(), batch_size=1, dtype=tf.float32)

        normalized_im = layers.Lambda(
            normalize_im,
            output_shape=self.im_obs_shape
        )([in_img_obs, im_mu, im_std])

        target = build_rnd_forward_model(self.im_obs_shape, self.hparams.rnd_emb_size,
                                         self.hparams.activation, name="target")
        target.trainable = False
        student = build_rnd_forward_model(self.im_obs_shape, self.hparams.rnd_emb_size,
                                          self.hparams.activation, name="student")

        y_target = target((normalized_im, dir_input))
        y_student = student((normalized_im, dir_input))
        rnd_error = layers.Lambda(
            lambda x: tf.reduce_mean(
                tf.square(x[0] - x[1]),
                axis=-1
            )
        )([y_target, y_student])
        return keras.Model(inputs=(in_img_obs, dir_input, im_mu, im_std), outputs=rnd_error)

    @staticmethod
    def HParams() -> type[RNDConvActorCriticHParams]:
        return RNDConvActorCriticHParams

    @staticmethod
    def model_id():
        return "rnd_conv_actor_critic"

from dataclasses import dataclass, field

import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from grid_exp_oe.models.base import RNNActorCriticBuilder, ModelHparams
from grid_exp_oe.common import nested_repeat
from grid_exp_oe.models.conv_actor_critic import configure_in_out_shapes, get_features_extractor


@dataclass
class LSTMConvActorCriticHParams(ModelHparams):
    policy_units: int = 32
    critic_units: int = 32
    lstm_unit: int = 32
    dense_layers: list[int] = field(default_factory=lambda: [32])
    attention: bool = True
    attention_units: int = 16
    activation: str = "relu"


def reset_lstm_states(previous_states: list[tf.Tensor], initial_states: list[tf.Tensor], starts: tf.Tensor):
    """
    Reset LSTM states based on a boolean mask.
     * previous_states: list [h, c] tensors, each with shape (batch_size, num_units)
     * initial_states: List of [h, c] tensors, each with shape (num_units,) or (batch_size, num_units)
     * starts: bool (batch_size, 1), True indicates that states must be reset
    
    """
    h_prev, c_prev = previous_states
    h_init, c_init = initial_states
    
    batch_size = tf.shape(h_prev)[0]
    
    # If initial states don't have batch dimension, expand and tile them
    if len(h_init.shape) == 1:
        h_init = tf.expand_dims(h_init, axis=0)
        c_init = tf.expand_dims(c_init, axis=0)
        
        h_init = tf.tile(h_init, [batch_size, 1])
        c_init = tf.tile(c_init, [batch_size, 1])
    
    # Broadcast the expanded starts tensor to match the shape of the state tensors
    # From (batch_size, 1) to (batch_size, num_units)
    mask = tf.broadcast_to(starts, tf.shape(h_prev))
    
    h_new = tf.where(mask, h_init, h_prev)
    c_new = tf.where(mask, c_init, c_prev)

    return [h_new, c_new]


class ResetLSTMState(layers.Layer):
    def __init__(self,lstm_units: int):
        super().__init__()
        self.lstm_units = lstm_units
        self.init_states =  [
            tf.zeros((lstm_units,), dtype=tf.float32),
            tf.zeros((lstm_units,), dtype=tf.float32),
        ]

    def call(self, inputs):
        previous_states, starts = inputs
        return reset_lstm_states(previous_states, self.init_states, starts)

    def get_config(self):
        return {"lstm_units", self.lstm_units}


class LSTMConvActorCriticBuilder(RNNActorCriticBuilder[LSTMConvActorCriticHParams]):
    def __init__(self, hparams: LSTMConvActorCriticHParams):
        self.hparams = hparams

        self.im_obs_shape = None
        self.num_actions = None

    def _build(self, env: gym.Env | gym.vector.VectorEnv, specs: dict | None = None) -> keras.Model:
        if specs is not None:
            self.im_obs_shape, self.num_actions = specs["im_obs_shape"], specs["num_actions"]
        else:
            if env is None:
                raise ValueError("if specs are not provided an env is required!")
            self.im_obs_shape, self.num_actions = configure_in_out_shapes(env)

        hparams: LSTMConvActorCriticHParams = self.hparams

        feature_extractor = get_features_extractor(self.im_obs_shape, hparams)
        in_img_obs = layers.Input(shape=self.im_obs_shape, dtype=tf.float32)
        dir_input = layers.Input(shape=(), dtype=tf.int32)
        previous_lstm_states = [
            layers.Input(shape=(hparams.lstm_unit,), dtype=tf.float32, name="h_state"),
            layers.Input(shape=(hparams.lstm_unit,), dtype=tf.float32, name="c_state")
        ]
        starts = layers.Input(shape=(1,), dtype=tf.bool, name="starts")

        states = ResetLSTMState(hparams.lstm_unit)((previous_lstm_states, starts))

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

        lstm_cell = layers.LSTMCell(hparams.lstm_unit)
        h, next_states = lstm_cell(x, states=states)

        z_policy = layers.Dense(hparams.policy_units, activation=hparams.activation)(h)
        logits = layers.Dense(self.num_actions, activation=None)(z_policy)

        z_critic = layers.Dense(hparams.critic_units, activation=hparams.activation)(h)
        value = layers.Dense(1, activation=None)(z_critic)

        return ((in_img_obs, dir_input), previous_lstm_states, starts), (logits, value, next_states)

    def initial_states(self, num_envs: int | None = None) -> list[tf.Tensor]:
        init_states =  [
            tf.zeros((self.hparams.lstm_unit,), dtype=tf.float32),
            tf.zeros((self.hparams.lstm_unit,), dtype=tf.float32),
        ]
        if num_envs is not None:
            return nested_repeat(init_states, num_envs)
        return init_states

    @staticmethod
    def HParams() -> type[LSTMConvActorCriticHParams]:
        return LSTMConvActorCriticHParams

    @staticmethod
    def model_id():
        return "lstm_conv_actor_critic"

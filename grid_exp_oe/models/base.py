from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
import gymnasium as gym
from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np
from tensorflow import keras
import tensorflow as tf

from grid_exp_oe.common import sample_logits, expand_batch_rec


@dataclass
class ModelHparams:
    ...


H = TypeVar("H", bound=ModelHparams)


class PolicyType(StrEnum):
    ACTOR_CRITIC = "actor_critic"
    VALUE_BASED = "value_based"
    RNN_ACTOR_CRITIC = "rnn_actor_critic"


class ModelBuilder(ABC, Generic[H]):
    @abstractmethod
    def __init__(self, hparams: H):
        self.hparams = hparams

    @abstractmethod
    def build(self, env: gym.Env | gym.vector.VectorEnv) -> keras.Model:
        """Builds the model for training purposes"""
        ...

    @abstractmethod
    def adapt_eval(self, model: keras.Model, expand_batch=False, return_numpy=False) -> Callable:
        """Adapt the policy for evaluation so it generates just the action (not logits nor values),
        sampling should be performed for random policies or greedy actions taken for
        value based."""
        ...

    @staticmethod
    @abstractmethod
    def model_id() -> str:
        ...

    @staticmethod
    @abstractmethod
    def HParams() -> type[H]:
        ...

    @staticmethod
    @abstractmethod
    def policy_type() -> PolicyType:
        """Supported policy type."""
        ...


class ActorCriticBuilder(ModelBuilder[H], Generic[H]):
    def build(self, env: gym.Env | gym.vector.VectorEnv) -> keras.Model:
        input_layers, (logits, value) = self._build(env)
        policy = keras.Model(inputs=(input_layers), outputs=(logits, value))
        return policy

    @abstractmethod
    def _build(self, env: gym.Env | gym.vector.VectorEnv) -> keras.Model:
        ...

    @staticmethod
    def policy_type() -> PolicyType:
        return PolicyType.ACTOR_CRITIC

    def adapt_eval(self, model: keras.Model, num_envs: int, expand_batch: bool, return_numpy=False) -> Callable:
        def policy(obs, start):
            if expand_batch:
                obs = expand_batch_rec(obs)
            logits, _ = model(obs)
            action = sample_logits(logits)
            if return_numpy:
                return action.numpy()
            return action
        return policy


def prepare_rnn_inputs(obs: np.ndarray | tuple[np.ndarray, ...], states: tf.Tensor | list[tf.Tensor], starts: np.ndarray):
    """Convert all rnn policy inputs to tf.Tensor and creates a proper dictionary with all them"""
    return {
            "observations": tuple(tf.convert_to_tensor(ob_) for ob_ in obs) if isinstance(obs, tuple) else tf.convert_to_tensor(obs),
            "previous_states": states,
            "starts": tf.convert_to_tensor(starts)
    }


class RNNActorCriticBuilder(ModelBuilder[H], Generic[H]):
    def build(self, env: gym.Env | gym.vector.VectorEnv | None = None, specs: dict | None = None) -> keras.Model:
        (input_layers, previous_states, starts), (logits, value, states) = self._build(env, specs)
        
        policy = keras.Model(
            inputs={
                "observations": input_layers,
                "previous_states": previous_states,
                "starts": starts
                },
            outputs=(logits, value, states)
        )
        return policy

    @abstractmethod
    def _build(self, env: gym.Env | gym.vector.VectorEnv, specs: dict | None = None):
        ...

    @abstractmethod
    def initial_states(self, num_envs: int | None = None) -> tf.Tensor | list[tf.Tensor]:
        ...

    @staticmethod
    def policy_type() -> PolicyType:
        return PolicyType.RNN_ACTOR_CRITIC

    def adapt_eval(self, model: keras.Model, num_envs: int, expand_batch: bool, return_numpy=False) -> Callable:
        states = None
        def policy(obs, start):
            nonlocal states
            if states is None:
                states = self.initial_states(num_envs)

            if expand_batch:
                obs = expand_batch_rec(obs)
                start = np.expand_dims(start, axis=0)
            logits, _, states = model(prepare_rnn_inputs(obs, states, start))
            action = sample_logits(logits)
            if return_numpy:
                return action.numpy()
            return action
        return policy

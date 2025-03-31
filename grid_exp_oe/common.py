import numpy as np
import tensorflow as tf


def sample_logits(logits):
    # https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    u = tf.random.uniform(tf.shape(logits))
    return tf.argmax(
            logits - tf.math.log(-tf.math.log(u)),
            axis=-1,
            output_type=tf.int32)


def expand_batch_rec(x: tf.Tensor | tuple):
    if isinstance(x, tuple):
        return tuple([expand_batch_rec(el) for el in x])
    return tf.expand_dims(x, axis=0)


def vectorized_returns(rewards: np.ndarray, terminal, gamma: float):
    returns = np.zeros_like(rewards)
    num_steps = rewards.shape[1]
    for i in range(num_steps - 1, -1, -1):
        if i == (num_steps - 1):
            returns[i] = rewards[i]
        else:
            returns[i] = rewards[i] + gamma * (1 - terminal[i]) * returns[i + 1]
    return returns


def get_gae_estimator(gamma: float, gae_lambda: float):
    def estimate_gae(estimated_value, next_estimated_value, reward, terminal):
        delta = reward + (1 - terminal) * gamma * next_estimated_value - estimated_value
        gae = np.empty((len(delta), 1), dtype=np.float32)
        for i in range(len(gae) - 1, -1, -1):
            if i == (len(gae) - 1):
                gae[i] = delta[i]
            else:
                gae[i] = delta[i] + gae_lambda * gamma * (1 - terminal[i]) * gae[i + 1]
        return gae
    return estimate_gae


def vectorize_gae_estimator(gae_estimator):
    def estimate(estimated_value, next_estimated_value, reward, terminal):
        return np.concatenate(
            [
                np.expand_dims(gae_estimator(
                    estimated_value[:, i],
                    next_estimated_value[:, i],
                    reward[:, i],
                    terminal[:, i]
                    ), axis=1) for i in range(estimated_value.shape[1])
            ],
            axis=1
        )
    return estimate


def nested_repeat(x, n):
    if isinstance(x, list):
        return [nested_repeat(x_, n) for x_ in x]
    return tf.repeat(tf.expand_dims(x, axis=0), n, axis=0)

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

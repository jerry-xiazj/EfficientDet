import tensorflow as tf


def drop_connect(inputs, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob + tf.random.uniform(
        [batch_size, 1, 1, 1],
        dtype=inputs.dtype
    )
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = tf.divide(inputs, survival_prob) * binary_tensor
    return output

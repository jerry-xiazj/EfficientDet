import tensorflow as tf
from typing import Text, Tuple, Union
import math


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob + tf.random.uniform([batch_size, 1, 1, 1],
                                                      dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = tf.divide(inputs, survival_prob) * binary_tensor
    return output


def activation_fn(features: tf.Tensor, act_type: Text):
    """Customized non-linear activation type."""
    if act_type == 'swish':
        return tf.nn.swish(features)
    elif act_type == 'swish_native':
        return features * tf.sigmoid(features)
    elif act_type == 'relu':
        return tf.nn.relu(features)
    elif act_type == 'relu6':
        return tf.nn.relu6(features)
    else:
        raise ValueError('Unsupported act_type {}'.format(act_type))


def batch_norm_act(inputs,
                   is_training_bn: bool,
                   act_type: Union[Text, None],
                   init_zero: bool = False,
                   data_format: Text = 'channels_last',
                   momentum: float = 0.99,
                   epsilon: float = 1e-3,
                   name: Text = None):
    """Performs a batch normalization followed by a non-linear activation.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    act_type: non-linear relu function type. If None, omits the relu operation.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    if data_format == 'channels_first':
        axis = 1
    else:
        axis = 3

    inputs = tf.keras.layers.BatchNormalization(
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        training=is_training_bn,
        gamma_initializer=gamma_initializer,
        name=name)

    if act_type:
        inputs = activation_fn(inputs, act_type)
    return inputs


def get_feat_sizes(image_size: Tuple[int, int],
                   max_level: int):
    """Get feat widths and heights for all levels.

  Args:
    image_size: A tuple (H, W).
    max_level: maximum feature level.

  Returns:
    feat_sizes: a list of tuples (height, width) for each level.
  """
    feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
    feat_size = image_size
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
    return feat_sizes

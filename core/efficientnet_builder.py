from core import EfficientNet
import tensorflow as tf


_DEFAULT_BLOCKS_ARGS = [
    EfficientNet.BlockArgs(
        kernel_size=2, num_repeat=1, input_filters=32, output_filters=16,
        expand_ratio=1, strides=[1, 1], se_ratio=0.25),
    EfficientNet.BlockArgs(
        kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
        expand_ratio=6, strides=[2, 2], se_ratio=0.25),
    EfficientNet.BlockArgs(
        kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
        expand_ratio=6, strides=[2, 2], se_ratio=0.25),
    EfficientNet.BlockArgs(
        kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
        expand_ratio=6, strides=[2, 2], se_ratio=0.25),
    EfficientNet.BlockArgs(
        kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
        expand_ratio=6, strides=[1, 1], se_ratio=0.25),
    EfficientNet.BlockArgs(
        kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
        expand_ratio=6, strides=[2, 2], se_ratio=0.25),
    EfficientNet.BlockArgs(
        kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
        expand_ratio=6, strides=[1, 1], se_ratio=0.25)
]

params_dict = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
}


def build_model(images, name, training, features_only=False, pooled_features_only=False):
    assert name in params_dict
    (width_coefficient, depth_coefficient, resolution, dropout_rate) = params_dict[name]
    global_params = EfficientNet.GlobalParams(
        blocks_args=_DEFAULT_BLOCKS_ARGS,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        survival_prob=0.8,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.swish,
        batch_norm=tf.keras.layers.BatchNormalization,
        use_se=True
    )
    model = EfficientNet.Model(_DEFAULT_BLOCKS_ARGS, global_params, name)
    outputs = model(
        images,
        training=training,
        features_only=features_only,
        pooled_features_only=pooled_features_only
    )
    if features_only:
        outputs = tf.identity(outputs, 'features')
    elif pooled_features_only:
        outputs = tf.identity(outputs, 'pooled_features')
    else:
        outputs = tf.identity(outputs, 'logits')
    return outputs, model.endpoints

from core import EfficientNet
from core import BiFPN
import tensorflow as tf

_DEFAULT_BLOCKS_ARGS = [
    EfficientNet.BlockArgs(kernel_size=2,
                           num_repeat=1,
                           input_filters=32,
                           output_filters=16,
                           expand_ratio=1,
                           strides=[1, 1],
                           se_ratio=0.25),
    EfficientNet.BlockArgs(kernel_size=3,
                           num_repeat=2,
                           input_filters=16,
                           output_filters=24,
                           expand_ratio=6,
                           strides=[2, 2],
                           se_ratio=0.25),
    EfficientNet.BlockArgs(kernel_size=5,
                           num_repeat=2,
                           input_filters=24,
                           output_filters=40,
                           expand_ratio=6,
                           strides=[2, 2],
                           se_ratio=0.25),
    EfficientNet.BlockArgs(kernel_size=3,
                           num_repeat=3,
                           input_filters=40,
                           output_filters=80,
                           expand_ratio=6,
                           strides=[2, 2],
                           se_ratio=0.25),
    EfficientNet.BlockArgs(kernel_size=5,
                           num_repeat=3,
                           input_filters=80,
                           output_filters=112,
                           expand_ratio=6,
                           strides=[1, 1],
                           se_ratio=0.25),
    EfficientNet.BlockArgs(kernel_size=5,
                           num_repeat=4,
                           input_filters=112,
                           output_filters=192,
                           expand_ratio=6,
                           strides=[2, 2],
                           se_ratio=0.25),
    EfficientNet.BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=192,
                           output_filters=320,
                           expand_ratio=6,
                           strides=[1, 1],
                           se_ratio=0.25)
]

efficientnet_model_param_dict = {
    # (width_coefficient, depth_coefficient, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 0.2),
    'efficientnet-b1': (1.0, 1.1, 0.2),
    'efficientnet-b2': (1.1, 1.2, 0.3),
    'efficientnet-b3': (1.2, 1.4, 0.3),
    'efficientnet-b4': (1.4, 1.8, 0.4),
    'efficientnet-b5': (1.6, 2.2, 0.4),
    'efficientnet-b6': (1.8, 2.6, 0.5),
    'efficientnet-b7': (2.0, 3.1, 0.5),
    'efficientnet-b8': (2.2, 3.6, 0.5),
    'efficientnet-l2': (4.3, 5.3, 0.5),
}

efficientdet_model_param_dict = {
    'efficientdet-d0':
    dict(
        name='efficientdet-d0',
        backbone_name='efficientnet-b0',
        image_size=512,
        fpn_num_filters=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
    ),
    'efficientdet-d1':
    dict(
        name='efficientdet-d1',
        backbone_name='efficientnet-b1',
        image_size=640,
        fpn_num_filters=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
    ),
    'efficientdet-d2':
    dict(
        name='efficientdet-d2',
        backbone_name='efficientnet-b2',
        image_size=768,
        fpn_num_filters=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
    ),
    'efficientdet-d3':
    dict(
        name='efficientdet-d3',
        backbone_name='efficientnet-b3',
        image_size=896,
        fpn_num_filters=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
    ),
    'efficientdet-d4':
    dict(
        name='efficientdet-d4',
        backbone_name='efficientnet-b4',
        image_size=1024,
        fpn_num_filters=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
    ),
    'efficientdet-d5':
    dict(
        name='efficientdet-d5',
        backbone_name='efficientnet-b5',
        image_size=1280,
        fpn_num_filters=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
    ),
    'efficientdet-d6':
    dict(
        name='efficientdet-d6',
        backbone_name='efficientnet-b6',
        image_size=1280,
        fpn_num_filters=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
    ),
    'efficientdet-d7':
    dict(
        name='efficientdet-d7',
        backbone_name='efficientnet-b6',
        image_size=1536,
        fpn_num_filters=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=5.0,
        fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
    ),
}


def build_backbone(images, name, training):
    (width_coefficient, depth_coefficient, dropout_rate) = efficientnet_model_param_dict[name]
    survival_prob = 1.0 if 'b0' in name else 0.8
    global_params = EfficientNet.GlobalParams(
        blocks_args=_DEFAULT_BLOCKS_ARGS,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        survival_prob=survival_prob,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.swish,
        batch_norm=tf.keras.layers.BatchNormalization,
        use_se=True)
    model = EfficientNet.Model(_DEFAULT_BLOCKS_ARGS, global_params, backbone_name)
    _ = model(images, training=training, features_only=True, pooled_features_only=False)
    features = {
        0: images,
        1: model.endpoints['reduction_1'],
        2: model.endpoints['reduction_2'],
        3: model.endpoints['reduction_3'],
        4: model.endpoints['reduction_4'],
        5: model.endpoints['reduction_5']
    }
    return features

def build_model(images,
                name,
                config,
                features_only=False,
                pooled_features_only=False,
                **kwargs):
    if kwargs:
        config.override(kwargs)

    logging.info(config)

    # build backbone features.
    backbone_name = efficientdet_model_param_dict[name]['backbone_name']
    features = build_backbone(images, backbone_name, config.is_training_bn)

    # build feature network.
    fpn_feats = BiFPN.build_feature_network(features, config)

    # build class and box predictions.
    class_outputs, box_outputs = BiFPN.build_class_and_box_outputs(fpn_feats, config)

  return class_outputs, box_outputs

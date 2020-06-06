import utils
import itertools
from hparams import Config
from absl import logging
import tensorflow as tf


def resample_feature_map(feat,
                         name,
                         target_height,
                         target_width,
                         target_num_channels,
                         is_training=True):
    """Resample input feature map to have target number of channels and size."""
    _, height, width, num_channels = feat.get_shape().as_list()

    if height is None or width is None or num_channels is None:
        raise ValueError(
            'shape[1] or shape[2] or shape[3] of feat is None (shape:{}).'.
            format(feat.shape))

    def _maybe_apply_1x1(feat):
        """Apply 1x1 conv to change layer width if necessary."""
        if num_channels != target_num_channels:
            feat = tf.keras.layers.Conv2D(feat,
                                          filters=target_num_channels,
                                          kernel_size=(1, 1),
                                          padding='same')
            feat = utils.batch_norm_act(feat,
                                        is_training_bn=is_training,
                                        act_type=None,
                                        name='bn')
        return feat

    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if height > target_height and width > target_width:
        feat = _maybe_apply_1x1(feat)
        height_stride_size = int((height - 1) // target_height + 1)
        width_stride_size = int((width - 1) // target_width + 1)
        feat = tf.keras.layers.MaxPooling2D(
            inputs=feat,
            pool_size=[height_stride_size + 1, width_stride_size + 1],
            strides=[height_stride_size, width_stride_size],
            padding='SAME')
    elif height <= target_height and width <= target_width:
        feat = _maybe_apply_1x1(feat)
        if height < target_height or width < target_width:
            feat = tf.image.resize(
                feat, [target_height, target_width],
                tf.image.ReszieMethod.NEAREST_NEIGHBOR)
    else:
        raise ValueError(
            'Incompatible target feature map size: target_height: {},'
            'target_width: {}'.format(target_height, target_width))

    return feat


def bifpn_dynamic_config(min_level, max_level):
    """A dynamic bifpn config that can adapt to different min/max levels."""
    config = Config()

    # Node id starts from the input features and monotonically increase whenever
    # a new node is added. Here is an example for level P3 - P7:
    #     P7 (4)              P7" (12)
    #     P6 (3)    P6' (5)   P6" (11)
    #     P5 (2)    P5' (6)   P5" (10)
    #     P4 (1)    P4' (7)   P4" (9)
    #     P3 (0)              P3" (8)
    # So output would be like:
    # [
    #   {'feat_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
    #   {'feat_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
    #   {'feat_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
    #   {'feat_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
    #   {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
    #   {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
    #   {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
    #   {'feat_level': 7, 'inputs_offsets': [4, 11]},  # for P7"
    # ]
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    def level_last_id(level):
        return node_ids[level][-1]

    def level_all_ids(level):
        return node_ids[level]

    id_cnt = itertools.count(num_levels)

    config.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        config.nodes.append({
            'feat_level':
            i,
            'inputs_offsets': [level_last_id(i),
                               level_last_id(i + 1)]
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        config.nodes.append({
            'feat_level':
            i,
            'inputs_offsets':
            level_all_ids(i) + [level_last_id(i - 1)]
        })
        node_ids[i].append(next(id_cnt))

    return config


def build_bifpn_layer(feats, feat_sizes, config):
    """Builds a feature pyramid given previous feature pyramid and config.

    Args:
    feats: [P3, P4, P5, P6, P7]
    config: a dict-like config, including all parameters.

    Returns:
        {3:P3", 4:P4", 5:P5", 6:P6", 7:P7"}
    """
    if config.fpn_config:
        fpn_config = config.fpn_config
    else:
        fpn_config = bifpn_dynamic_config(config.min_level, config.max_level, config.fpn_weight_method)

    num_output_connections = [0 for _ in feats]
    for i, fnode in enumerate(fpn_config.nodes):
        logging.info('fnode %d : %s', i, fnode)
        new_node_height = feat_sizes[fnode['feat_level']]['height']
        new_node_width = feat_sizes[fnode['feat_level']]['width']
        nodes = []
        for idx, input_offset in enumerate(fnode['inputs_offsets']):
            input_node = feats[input_offset]
            num_output_connections[input_offset] += 1
            input_node = resample_feature_map(
                input_node,
                '{}_{}_{}'.format(idx, input_offset, len(feats)),
                new_node_height,
                new_node_width,
                config.fpn_num_filters,
                config.is_training_bn)
            nodes.append(input_node)

        # Combine all nodes.
        dtype = nodes[0].dtype
        edge_weights = [
            tf.nn.relu(
                tf.cast(tf.Variable(1.0, name='WSM'), dtype=dtype))
            for _ in range(len(fnode['inputs_offsets']))
        ]
        weights_sum = tf.add_n(edge_weights)
        nodes = [
            nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
            for i in range(len(nodes))
        ]
        new_node = tf.add_n(nodes)

        new_node = utils.activation_fn(new_node, config.act_type)

        new_node = tf.keras.layers.SeparableConv2D(
            new_node,
            filters=config.fpn_num_filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True,
            depth_multiplier=1,
            name='conv')

        new_node = tf.keras.layers.BatchNormalization(
            inputs=new_node,
            training=config.is_training_bn,
            name='bn')

        feats.append(new_node)
        num_output_connections.append(0)

    output_feats = {}
    for level in range(config.min_level, config.max_level + 1):
        for i, fnode in enumerate(reversed(fpn_config.nodes)):
            if fnode['feat_level'] == level:
                output_feats[level] = feats[-1 - i]
                break
    return output_feats


def build_feature_network(features, config):
    """Build FPN input features.

    Args:
    features: {0:image, 1:endpoints["reduction_1], ...}
    config: a dict-like config, including all parameters.

    Returns:
        A dict from levels to the feature maps processed after feature network.
    """
    feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
    feats = []
    if config.min_level not in features.keys():
        raise ValueError(
            'features.keys ({}) should include min_level ({})'.format(
                features.keys(), config.min_level))

    # Build additional input features that are not from backbone.
    for level in range(config.min_level, config.max_level + 1):
        if level in features.keys():
            feats.append(features[level])
        else:
            # Adds a coarser level by downsampling the last feature map.
            feats.append(
                resample_feature_map(
                    feats[-1],
                    name='p%d' % level,
                    target_height=(feats[-1].shape[1] - 1) // 2 + 1,
                    target_width=(feats[-1].shape[2] - 1) // 2 + 1,
                    target_num_channels=config.fpn_num_filters,
                    is_training=config.is_training_bn
                )
            )

    for rep in range(config.fpn_cell_repeats):
        logging.info('building cell %d', rep)
        new_feats = build_bifpn_layer(feats, feat_sizes, config)

        feats = [
            new_feats[level]
            for level in range(config.min_level, config.max_level + 1)
        ]

    return new_feats

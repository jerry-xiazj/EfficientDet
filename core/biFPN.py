import utils
import functools
import itertools
import hparams_config
from absl import logging
import tensorflow as tf


def resample_feature_map(feat,
                         name,
                         target_height,
                         target_width,
                         target_num_channels,
                         apply_bn=False,
                         is_training=None,
                         conv_after_downsample=False,
                         use_native_resize_op=False,
                         pooling_type=None,
                         data_format='channels_last'):
    """Resample input feature map to have target number of channels and size."""
    if data_format == 'channels_first':
        _, num_channels, height, width = feat.get_shape().as_list()
    else:
        _, height, width, num_channels = feat.get_shape().as_list()

    if height is None or width is None or num_channels is None:
        raise ValueError(
            'shape[1] or shape[2] or shape[3] of feat is None (shape:{}).'.
            format(feat.shape))
    if apply_bn and is_training is None:
        raise ValueError('If BN is applied, need to provide is_training')

    def _maybe_apply_1x1(feat):
        """Apply 1x1 conv to change layer width if necessary."""
        if num_channels != target_num_channels:
            feat = tf.keras.layers.Conv2D(feat,
                                          filters=target_num_channels,
                                          kernel_size=(1, 1),
                                          padding='same',
                                          data_format=data_format)
            if apply_bn:
                feat = utils.batch_norm_act(feat,
                                            is_training_bn=is_training,
                                            act_type=None,
                                            data_format=data_format,
                                            name='bn')
        return feat

    with tf.variable_scope('resample_{}'.format(name)):
        # If conv_after_downsample is True, when downsampling, apply 1x1 after
        # downsampling for efficiency.
        if height > target_height and width > target_width:
            if not conv_after_downsample:
                feat = _maybe_apply_1x1(feat)
            height_stride_size = int((height - 1) // target_height + 1)
            width_stride_size = int((width - 1) // target_width + 1)
            if pooling_type == 'max' or pooling_type is None:
                # Use max pooling in default.
                feat = tf.keras.layers.MaxPooling2D(
                    inputs=feat,
                    pool_size=[height_stride_size + 1, width_stride_size + 1],
                    strides=[height_stride_size, width_stride_size],
                    padding='SAME',
                    data_format=data_format)
            elif pooling_type == 'avg':
                feat = tf.keras.layers.AveragePooling2D(
                    inputs=feat,
                    pool_size=[height_stride_size + 1, width_stride_size + 1],
                    strides=[height_stride_size, width_stride_size],
                    padding='SAME',
                    data_format=data_format)
            else:
                raise ValueError(
                    'Unknown pooling type: {}'.format(pooling_type))
            if conv_after_downsample:
                feat = _maybe_apply_1x1(feat)
        elif height <= target_height and width <= target_width:
            feat = _maybe_apply_1x1(feat)
            if height < target_height or width < target_width:
                height_scale = target_height // height
                width_scale = target_width // width
                if (use_native_resize_op or target_height % height != 0
                        or target_width % width != 0):
                    if data_format == 'channels_first':
                        feat = tf.transpose(feat, [0, 2, 3, 1])
                    feat = tf.image.resize(
                        feat, [target_height, target_width],
                        tf.image.ReszieMethod.NEAREST_NEIGHBOR)
                    if data_format == 'channels_first':
                        feat = tf.transpose(feat, [0, 3, 1, 2])
                else:
                    feat = nearest_upsampling(feat,
                                              height_scale=height_scale,
                                              width_scale=width_scale,
                                              data_format=data_format)
        else:
            raise ValueError(
                'Incompatible target feature map size: target_height: {},'
                'target_width: {}'.format(target_height, target_width))

    return feat


def nearest_upsampling(data, height_scale, width_scale, data_format):
    """Nearest neighbor upsampling implementation."""
    with tf.name_scope('nearest_upsampling'):
        # Use reshape to quickly upsample the input. The nearest pixel is selected
        # implicitly via broadcasting.
        if data_format == 'channels_first':
            # Possibly faster for certain GPUs only.
            bs, c, h, w = data.get_shape().as_list()
            bs = -1 if bs is None else bs
            data = tf.reshape(data, [bs, c, h, 1, w, 1]) * tf.ones(
                [1, 1, 1, height_scale, 1, width_scale], dtype=data.dtype)
            return tf.reshape(data, [bs, c, h * height_scale, w * width_scale])

        # Normal format for CPU/TPU/GPU.
        bs, h, w, c = data.get_shape().as_list()
        bs = -1 if bs is None else bs
        data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
            [1, 1, height_scale, 1, width_scale, 1], dtype=data.dtype)
        return tf.reshape(data, [bs, h * height_scale, w * width_scale, c])


def bifpn_dynamic_config(min_level, max_level, weight_method):
    """A dynamic bifpn config that can adapt to different min/max levels."""
    p = hparams_config.Config()
    p.weight_method = weight_method or 'fastattn'

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

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        p.nodes.append({
            'feat_level':
            i,
            'inputs_offsets': [level_last_id(i),
                               level_last_id(i + 1)]
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        p.nodes.append({
            'feat_level':
            i,
            'inputs_offsets':
            level_all_ids(i) + [level_last_id(i - 1)]
        })
        node_ids[i].append(next(id_cnt))

    return p


def build_bifpn_layer(feats, feat_sizes, config):
    """Builds a feature pyramid given previous feature pyramid and config."""
    p = config  # use p to denote the network config.
    if p.fpn_config:
        fpn_config = p.fpn_config
    else:
        fpn_config = bifpn_dynamic_config(p.min_level, p.max_level, p.fpn_weight_method)

    num_output_connections = [0 for _ in feats]
    for i, fnode in enumerate(fpn_config.nodes):
        with tf.variable_scope('fnode{}'.format(i)):
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
                    p.fpn_num_filters,
                    p.apply_bn_for_resampling,
                    p.is_training_bn,
                    p.conv_after_downsample,
                    p.use_native_resize_op,
                    p.pooling_type,
                    data_format=config.data_format)
                nodes.append(input_node)

            # Combine all nodes.
            dtype = nodes[0].dtype
            if fpn_config.weight_method == 'attn':
                edge_weights = [
                    tf.cast(tf.Variable(1.0, name='WSM'), dtype=dtype)
                    for _ in range(len(fnode['inputs_offsets']))
                ]
                normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
                nodes = tf.stack(nodes, axis=-1)
                new_node = tf.reduce_sum(
                    tf.multiply(nodes, normalized_weights), -1)
            elif fpn_config.weight_method == 'fastattn':
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
            elif fpn_config.weight_method == 'sum':
                new_node = tf.add_n(nodes)
            else:
                raise ValueError('unknown weight_method {}'.format(
                    fpn_config.weight_method))

            with tf.variable_scope('op_after_combine{}'.format(len(feats))):
                if not p.conv_bn_act_pattern:
                    new_node = utils.activation_fn(new_node, p.act_type)

                if p.separable_conv:
                    conv_op = functools.partial(tf.keras.layers.SeparableConv2D,
                                                depth_multiplier=1)
                else:
                    conv_op = tf.keras.layers.Conv2D

                new_node = conv_op(
                    new_node,
                    filters=p.fpn_num_filters,
                    kernel_size=(3, 3),
                    padding='same',
                    use_bias=True if not p.conv_bn_act_pattern else False,
                    data_format=config.data_format,
                    name='conv')

                new_node = utils.batch_norm_act(
                    new_node,
                    is_training_bn=p.is_training_bn,
                    act_type=None if not p.conv_bn_act_pattern else p.act_type,
                    data_format=config.data_format,
                    use_tpu=p.use_tpu,
                    name='bn')

            feats.append(new_node)
            num_output_connections.append(0)

    output_feats = {}
    for l in range(p.min_level, p.max_level + 1):
        for i, fnode in enumerate(reversed(fpn_config.nodes)):
            if fnode['feat_level'] == l:
                output_feats[l] = feats[-1 - i]
                break
    return output_feats


def build_feature_network(features, config):
    """Build FPN input features.

  Args:
   features: input tensor.
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
            h_id, w_id = (2, 3) if config.data_format == 'channels_first' else (1, 2)
            # Adds a coarser level by downsampling the last feature map.
            feats.append(
                resample_feature_map(
                    feats[-1],
                    name='p%d' % level,
                    target_height=(feats[-1].shape[h_id] - 1) // 2 + 1,
                    target_width=(feats[-1].shape[w_id] - 1) // 2 + 1,
                    target_num_channels=config.fpn_num_filters,
                    apply_bn=config.apply_bn_for_resampling,
                    is_training=config.is_training_bn,
                    conv_after_downsample=config.conv_after_downsample,
                    use_native_resize_op=config.use_native_resize_op,
                    pooling_type=config.pooling_type,
                    data_format=config.data_format))

    with tf.variable_scope('fpn_cells'):
        for rep in range(config.fpn_cell_repeats):
            with tf.variable_scope('cell_{}'.format(rep)):
                logging.info('building cell %d', rep)
                new_feats = build_bifpn_layer(feats, feat_sizes, config)

                feats = [
                    new_feats[level]
                    for level in range(config.min_level, config.max_level + 1)
                ]

    return new_feats

# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

import collections
import tensorflow as tf
import core.utils as utils
from absl import logging


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum',
    'batch_norm_epsilon',
    'dropout_rate',
    'num_classes',
    'width_coefficient',
    'depth_coefficient',
    'depth_divisor',
    'min_depth',
    'survival_prob',
    'relu_fn',
    'use_se',
    'blocks_args'
])
GlobalParams.__new__.__defaults__ = (None, ) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

    Attributes:
        endpoints: dict. A list of internal tensors.
    """
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._has_se = (global_params.use_se
                        and self._block_args.se_ratio is not None
                        and 0 < self._block_args.se_ratio <= 1)
        self.endpoints = None
        self._kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_out', distribution='untruncated_normal')

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        kernel_size = self._block_args.kernel_size

        # Expansion phase.
        self._expand_conv = tf.keras.layers.Conv2D(
            name='conv2d',
            kernel_initializer=self._kernel_initializer,
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            use_bias=False
        )
        self._bn0 = tf.keras.layers.BatchNormalization(
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon
        )

        # Depth-wise convolution phase.
        self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            name='depthwise_conv2d',
            depthwise_initializer=self._kernel_initializer,
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides,
            padding='same',
            use_bias=False
        )
        self._bn1 = tf.keras.layers.BatchNormalization(
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon
        )

        if self._has_se:
            num_reduced_filters = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = tf.keras.layers.Conv2D(
                name='conv2d',
                kernel_initializer=self._kernel_initializer,
                filters=num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same',
                use_bias=True
            )
            self._se_expand = tf.keras.layers.Conv2D(
                name='conv2d',
                kernel_initializer=self._kernel_initializer,
                filters=filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same',
                use_bias=True
            )

        # Output phase.
        filters = self._block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            name='conv2d',
            kernel_initializer=self._kernel_initializer,
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            use_bias=False
        )
        self._bn2 = tf.keras.layers.BatchNormalization(
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon
        )

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.

        Args:
        input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

        Returns:
        A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(input_tensor, [1, 2], keepdims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        logging.info('Built Squeeze and Excitation with tensor shape: %s',
                     (se_tensor.shape))
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, survival_prob=None):
        """Implementation of call().

        Args:
        inputs: the inputs tensor.
        training: boolean, whether the model is constructed for training.
        survival_prob: float, between 0 to 1, drop connect rate.

        Returns:
        A output tensor.
        """
        logging.info('Block %s  input shape: %s', self.name, inputs.shape)
        x = inputs

        if self._block_args.expand_ratio != 1:
            x = self._relu_fn(self._bn0(self._expand_conv(x), training=training))
            logging.info('Expand shape: %s', x.shape)

        x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
        logging.info('DWConv shape: %s', x.shape)

        if self._has_se:
            x = self._call_se(x)

        self.endpoints = {'expansion_output': x}

        x = self._bn2(self._project_conv(x), training=training)
        # Add identity so that quantization-aware training can insert quantization
        # ops correctly.
        x = tf.identity(x)
        if all(
                s == 1 for s in self._block_args.strides
        ) and self._block_args.input_filters == self._block_args.output_filters:
            # Apply only if skip connection presents.
            if survival_prob:
                x = utils.drop_connect(x, training, survival_prob)
            x = tf.add(x, inputs)
        logging.info('Project shape: %s', x.shape)
        return x


class Model(tf.keras.Model):
    """A class implements tf.keras.Model for MNAS-like model.

        Reference: https://arxiv.org/abs/1807.11626
    """
    def __init__(self, blocks_args=None, global_params=None, name=None):
        """Initializes an `Model` instance.

        Args:
        blocks_args: A list of BlockArgs to construct block modules.
        global_params: GlobalParams, a set of global parameters.
        name: A string of layer name.

        Raises:
        ValueError: when blocks_args is not specified as a list.
        """
        super().__init__(name=name)
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon

        self.endpoints = None

        self._kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_out', distribution='untruncated_normal')
        self._dense_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0/3.0, mode='fan_out', distribution='uniform')
        self._build()

    def _build(self):
        """Builds a model."""
        self._blocks = []

        # Stem part.
        self._conv_stem = tf.keras.layers.Conv2D(
            name='conv2d',
            kernel_initializer=self._kernel_initializer,
            filters=utils.round_filters(32, self._global_params),
            kernel_size=[3, 3],
            strides=[2, 2],
            padding='same',
            use_bias=False)
        self._bn0 = tf.keras.layers.BatchNormalization(
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon
        )

        # Builds blocks.
        for i, block_args in enumerate(self._blocks_args):
            # Update block input and output filters based on depth multiplier.
            input_filters = utils.round_filters(block_args.input_filters, self._global_params)
            output_filters = utils.round_filters(block_args.output_filters, self._global_params)
            repeats = utils.round_repeats(block_args.num_repeat, self._global_params)
            block_args = block_args._replace(
                input_filters=input_filters,
                output_filters=output_filters,
                num_repeat=repeats
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))

            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                block_args = block_args._replace(
                    input_filters=block_args.output_filters,
                    strides=[1, 1]
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head part.
        self._conv_head = tf.keras.layers.Conv2D(
            name='conv2d',
            kernel_initializer=self._kernel_initializer,
            filters=utils.round_filters(1280, self._global_params),
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            use_bias=False
        )
        self._bn1 = tf.keras.layers.BatchNormalization(
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon
        )

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D()

        if self._global_params.num_classes:
            self._fc = tf.keras.layers.Dense(
                self._global_params.num_classes,
                kernel_initializer=self._dense_initializer)
        else:
            self._fc = None

        if self._global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self,
             inputs,
             training=True,
             features_only=False,
             pooled_features_only=False):
        """Implementation of call().

        Args:
        inputs: input tensors.
        training: boolean, whether the model is constructed for training.
        features_only: build the base feature network only.
        pooled_features_only: build the base network for features extraction
            (after 1x1 conv layer and global pooling, but before dropout and fc
            head).

        Returns:
        output tensors.
        """
        outputs = None
        self.endpoints = {}
        reduction_idx = 0

        # Calls Stem layers
        outputs = self._conv_stem(inputs)
        outputs = self._bn0(outputs, training=training)
        outputs = self._relu_fn(outputs)
        logging.info('Built stem layers with output shape: %s', outputs.shape)
        self.endpoints['stem'] = outputs

        # Calls blocks.
        for idx, block in enumerate(self._blocks):
            is_reduction = False  # reduction flag for blocks after the stem layer

            if ((idx == len(self._blocks) - 1)
                    or self._blocks[idx + 1].block_args().strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            survival_prob = self._global_params.survival_prob
            if survival_prob:
                drop_rate = 1.0 - survival_prob
                survival_prob = 1.0 - drop_rate * idx / len(self._blocks)
                logging.info('block_%s survival_prob: %s', idx, survival_prob)
            outputs = block.call(outputs, training=training, survival_prob=survival_prob)
            self.endpoints['block_%s' % idx] = outputs
            if is_reduction:
                self.endpoints['reduction_%s' % reduction_idx] = outputs
            if block.endpoints:
                for k, v in block.endpoints.items():
                    self.endpoints['block_%s/%s' % (idx, k)] = v
                    if is_reduction:
                        self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        self.endpoints['features'] = outputs

        if not features_only:
            # Calls final layers and returns logits.
            outputs = self._conv_head(outputs)
            outputs = self._bn1(outputs, training=training)
            outputs = self._relu_fn(outputs)
            self.endpoints['head_1x1'] = outputs

            outputs = self._avg_pooling(outputs)
            self.endpoints['pooled_features'] = outputs
            if not pooled_features_only:
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                self.endpoints['global_pool'] = outputs
                if self._fc:
                    outputs = self._fc(outputs)
                self.endpoints['head'] = outputs
        return outputs

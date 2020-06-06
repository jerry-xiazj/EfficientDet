# Copyright 2020 Google ResearcCFG. All Rights Reserved.
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
"""Hparams for model architecture and trainer."""

from easydict import EasyDict as edict


CFG = edict()

# model name.
CFG.name = 'efficientdet-d1'

# activation type: see activation_fn in utils.py.
CFG.act_type = 'swish'

# input preprocessing parameters
CFG.image_size = 640  # An integer or a string WxH such as 640x320.
CFG.input_rand_hflip = True
CFG.train_scale_min = 0.1
CFG.train_scale_max = 2.0
CFG.autoaugment_policy = None

# dataset specific parameters
CFG.num_classes = 90
CFG.skip_crowd_during_training = True
CFG.label_id_mapping = None

# model architecture
CFG.min_level = 3
CFG.max_level = 7
CFG.num_scales = 3
CFG.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
CFG.anchor_scale = 4.0
# is batchnorm training mode
CFG.is_training_bn = True
# optimization
CFG.momentum = 0.9
CFG.learning_rate = 0.08
CFG.lr_warmup_init = 0.008
CFG.lr_warmup_epoch = 1.0
CFG.first_lr_drop_epoch = 200.0
CFG.second_lr_drop_epoch = 250.0
CFG.poly_lr_power = 0.9
CFG.clip_gradients_norm = 10.0
CFG.num_epochs = 300

# classification loss
CFG.alpha = 0.25
CFG.gamma = 1.5
# localization loss
CFG.delta = 0.1
CFG.box_loss_weight = 50.0
# regularization l2 loss.
CFG.weight_decay = 4e-5
# enable bfloat
CFG.use_tpu = True
# precision: one of 'float32', 'mixed_float16', 'mixed_bfloat16'.
CFG.precision = None  # If None, use float32.

# For detection.
CFG.box_class_repeats = 3
CFG.fpn_cell_repeats = 3
CFG.fpn_num_filters = 88
CFG.separable_conv = True
CFG.apply_bn_for_resampling = True
CFG.conv_after_downsample = False
CFG.conv_bn_act_pattern = False
CFG.use_native_resize_op = False
CFG.pooling_type = None

# version.
CFG.fpn_name = None
CFG.fpn_config = None

# No stochastic depth in default.
CFG.survival_prob = None

CFG.lr_decay_method = 'cosine'
CFG.moving_average_decay = 0.9998
CFG.ckpt_var_scope = None  # ckpt variable scope.
# exclude vars when loading pretrained ckpts.
CFG.var_exclude_expr = '.*/class-predict/.*'  # exclude class weights in default

CFG.backbone_name = 'efficientnet-b1'
CFG.backbone_config = None
CFG.var_freeze_expr = None

# RetinaNet.
CFG.resnet_depth = 50


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


# def get_efficientdet_config(model_name='efficientdet-d1'):
#     """Get the default config for EfficientDet based on model name."""
#     h = default_detection_configs()
#     CFG.override(efficientdet_model_param_dict[model_name])
#     return h


# def get_detection_config(model_name):
#     if model_name.startswith('efficientdet'):
#         return get_efficientdet_config(model_name)
#     else:
#         raise ValueError('model name must start with efficientdet.')

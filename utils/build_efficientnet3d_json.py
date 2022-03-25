# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
"""EfficientNet V2 models for Keras.

Reference:
- [EfficientNetV2: Smaller Models and Faster Training](
    https://arxiv.org/abs/2104.00298) (ICML 2021)
"""

import copy
import math

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.applications import imagenet_utils
# from keras.engine import training
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import data_utils
from tensorflow.keras import utils
import tensorflow as tf

from utils.new_layers import DepthwiseConv3D


DEFAULT_BLOCKS_ARGS = {
    "efficientnetv2-s": [{
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 24,
        "output_filters": 24,
        "expand_ratio": 1,
        "se_ratio": 0.0,
        "strides": 1,
        "conv_type": 1,
    }, {
        "kernel_size": 3,
        "num_repeat": 4,
        "input_filters": 24,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0.0,
        "strides": 2,
        "conv_type": 1,
    }, {
        "conv_type": 1,
        "expand_ratio": 4,
        "input_filters": 48,
        "kernel_size": 3,
        "num_repeat": 4,
        "output_filters": 64,
        "se_ratio": 0,
        "strides": 2,
    }, {
        "conv_type": 0,
        "expand_ratio": 4,
        "input_filters": 64,
        "kernel_size": 3,
        "num_repeat": 6,
        "output_filters": 128,
        "se_ratio": 0.25,
        "strides": 2,
    }, {
        "conv_type": 0,
        "expand_ratio": 6,
        "input_filters": 128,
        "kernel_size": 3,
        "num_repeat": 9,
        "output_filters": 160,
        "se_ratio": 0.25,
        "strides": 1,
    }, {
        "conv_type": 0,
        "expand_ratio": 6,
        "input_filters": 160,
        "kernel_size": 3,
        "num_repeat": 15,
        "output_filters": 256,
        "se_ratio": 0.25,
        "strides": 2,
    }],
    "efficientnetv2-m": [
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-l": [
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b0": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b1": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b2": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b3": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal"
    }
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1. / 3.,
        "mode": "fan_out",
        "distribution": "uniform"
    }
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

  Reference:
  - [EfficientNetV2: Smaller Models and Faster Training](
      https://arxiv.org/abs/2104.00298) (ICML 2021)

  This function returns a Keras image classification model,
  optionally loaded with weights pre-trained on ImageNet.

  For image classification use cases, see
  [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: each Keras Application expects a specific kind of input preprocessing.
  For EfficientNetV2, by default input preprocessing is included as a part of the
  model (as a `Rescaling` layer), and thus
  `tf.keras.applications.efficientnet_v2.preprocess_input` is actually a
  pass-through function. In this use case, EfficientNetV2 models expect their inputs
  to be float tensors of pixels with values in the [0-255] range.
  At the same time, preprocessing as a part of the model (i.e. `Rescaling`
  layer) can be disabled by setting `include_preprocessing` argument to False.
  With preprocessing disabled EfficientNetV2 models expect their inputs to be float
  tensors of pixels with values in the [-1, 1] range.

  Args:
    include_top: Boolean, whether to include the fully-connected
      layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
      `"imagenet"` (pre-training on ImageNet),
      or the path to the weights file to be loaded. Defaults to `"imagenet"`.
    input_tensor: Optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False.
      It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`. Defaults to None.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional layer.
      - `"avg"` means that global average pooling
          will be applied to the output of the
          last convolutional layer, and thus
          the output of the model will be a 3D tensor.
      - `"max"` means that global max pooling will
          be applied.
    classes: Optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified. Defaults to 1000 (number of
      ImageNet classes).
    classifier_activation: A string or callable. The activation function to use
      on the `"top"` layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
      Defaults to `"softmax"`.
      When loading pretrained weights, `classifier_activation` can only
      be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""


def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def MBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name=None,
):
    """MBConv block: Mobile Inverted Residual Bottleneck."""
    bn_axis = 4 if backend.image_data_format() == "channels_last" else 1

    if name is None:
        name = backend.get_uid("block0")

    def apply(inputs):
        res = []
        # expansion phase
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            res.extend(
                [{
                    'name': name + "expand_conv",
                    'class_name': 'Conv3D',
                    'config': {
                        'filters': filters,
                        'kernel_size': 1,
                        'strides': 1,
                        'kernel_initializer': CONV_KERNEL_INITIALIZER,
                        'padding': "same",
                        'data_format': "channels_last",
                        'use_bias': False,

                    }
                },
                    {
                    'name': name + "expand_bn",
                    'class_name': 'BatchNormalization',
                    'config': {
                        'axis': bn_axis,
                        'momentum': bn_momentum,
                    }
                },
                    {
                    'name': name + "expand_activation",
                    'class_name': 'Activation',
                    'config': {
                        'activation': activation
                    }
                }]
            )

        # depthwise conv
        res.extend(
            [{
                'name': name + "dwconv2",
                'class_name': 'DepthwiseConv3D',
                'config': {
                    'kernel_size': kernel_size,
                    'strides': strides,
                    'depthwise_initializer': CONV_KERNEL_INITIALIZER,
                    'padding': "same",
                    'data_format': "channels_last",
                    'use_bias': False,

                }
            },
                {
                'name': name + "bn",
                'class_name': 'BatchNormalization',
                'config': {
                    'axis': bn_axis,
                    'momentum': bn_momentum,
                }
            },
                {
                'name': name + "activation",
                'class_name': 'Activation',
                'config': {
                    'activation': activation
                }
            }]
        )

        # squeeze & excitation
        if 0 < se_ratio <= 1:
            depthwise_output = res[-1]['name']
            filters_se = max(1, int(input_filters * se_ratio))
            res.append(
                {
                    'name': name + "se_squeeze",
                    'class_name': 'GlobalAveragePooling3D',
                    'inputs': [depthwise_output]
                }
            )
            if bn_axis == 1:
                se_shape = (filters, 1, 1, 1)
            else:
                se_shape = (1, 1, 1, filters)
            res.extend([
                {
                    'name': name + "se_reshape",
                    'class_name': 'Reshape',
                    'config': {
                        'target_shape': se_shape
                    }
                },
                {
                    'name': name + "se_reduce",
                    'class_name': 'Conv3D',
                    'config': {
                        'filters': filters_se,
                        'kernel_size': 1,
                        'padding': "same",
                        'activation': activation,
                        'kernel_initializer': CONV_KERNEL_INITIALIZER,
                    }
                },
                {
                    'name': name + "se_expand",
                    'class_name': 'Conv3D',
                    'config': {
                        'filters': filters,
                        'kernel_size': 1,
                        'padding': "same",
                        'activation': "sigmoid",
                        'kernel_initializer': CONV_KERNEL_INITIALIZER,
                    }
                }
            ])

            se_output_name = res[-1]['name']

            res.append(
                {
                    'name': name + "se_excite",
                    'class_name': 'Multiply',
                    'inputs': [depthwise_output, se_output_name]
                }
            )

        # output phase
        res.extend([
            {
                'name': name + "project_conv",
                'class_name': 'Conv3D',
                'config': {
                    'filters': output_filters,
                    'kernel_size': 1,
                    'strides': 1,
                    'kernel_initializer': CONV_KERNEL_INITIALIZER,
                    'padding': "same",
                    'data_format': "channels_last",
                    'use_bias': False,
                }
            },
            {
                'name': name + "project_bn",
                'class_name': 'BatchNormalization',
                'config': {
                    'axis': bn_axis,
                    'momentum': bn_momentum,
                }
            }
        ])

        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                res.append({
                    'name': name + 'drop',
                    'class_name': 'Dropout',
                    'config': {
                        'rate': survival_probability,
                        'noise_shape': (None, 1, 1, 1, 1),
                    }
                })
            last_layer_name = res[-1]['name']
            res.append({
                'name': name + 'add',
                'class_name': 'Add',
                'inputs': [
                    last_layer_name, inputs
                ]
            })

        return res

    return apply


def FusedMBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name=None,
):
    """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a conv3D."""
    bn_axis = 4 if backend.image_data_format() == "channels_last" else 1

    if name is None:
        name = backend.get_uid("block0")

    def apply(inputs):
        res = []
        # expansion phase
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            res.extend(
                [{
                    'name': name + "expand_conv",
                    'class_name': 'Conv3D',
                    'config': {
                        'filters': filters,
                        'kernel_size': kernel_size,
                        'strides': strides,
                        'kernel_initializer': CONV_KERNEL_INITIALIZER,
                        'padding': "same",
                        'data_format': "channels_last",
                        'use_bias': False,

                    }
                },
                    {
                    'name': name + "expand_bn",
                    'class_name': 'BatchNormalization',
                    'config': {
                        'axis': bn_axis,
                        'momentum': bn_momentum,
                    }
                },
                    {
                    'name': name + "expand_activation",
                    'class_name': 'Activation',
                    'config': {
                        'activation': activation
                    }
                }]
            )

        # squeeze & excitation
        if 0 < se_ratio <= 1:
            if len(res) > 0:
                depthwise_output = res[-1]['name']
            else:
                depthwise_output = inputs
            filters_se = max(1, int(input_filters * se_ratio))
            res.append(
                {
                    'name': name + "se_squeeze",
                    'class_name': 'GlobalAveragePooling3D',
                    'inputs': [depthwise_output]
                }
            )
            if bn_axis == 1:
                se_shape = (filters, 1, 1, 1)
            else:
                se_shape = (1, 1, 1, filters)
            res.extend([
                {
                    'name': name + "se_reshape",
                    'class_name': 'Reshape',
                    'config': {
                        'target_shape': se_shape
                    }
                },
                {
                    'name': name + "se_reduce",
                    'class_name': 'Conv3D',
                    'config': {
                        'filters': filters_se,
                        'kernel_size': 1,
                        'padding': "same",
                        'activation': activation,
                        'kernel_initializer': CONV_KERNEL_INITIALIZER,
                    }
                },
                {
                    'name': name + "se_expand",
                    'class_name': 'Conv3D',
                    'config': {
                        'filters': filters,
                        'kernel_size': 1,
                        'padding': "same",
                        'activation': "sigmoid",
                        'kernel_initializer': CONV_KERNEL_INITIALIZER,
                    }
                }
            ])

            se_output_name = res[-1]['name']

            res.append(
                {
                    'name': name + "se_excite",
                    'class_name': 'Multiply',
                    'inputs': [depthwise_output, se_output_name]
                }
            )

        # output phase
        res.extend([
            {
                'name': name + "project_conv",
                'class_name': 'Conv3D',
                'config': {
                    'filters': output_filters,
                    'kernel_size': 1 if expand_ratio != 1 else kernel_size,
                    'strides': 1 if expand_ratio != 1 else strides,
                    'kernel_initializer': CONV_KERNEL_INITIALIZER,
                    'padding': "same",
                    # 'data_format':"channels_last",
                    'use_bias': False,
                }
            },
            {
                'name': name + "project_bn",
                'class_name': 'BatchNormalization',
                'config': {
                    'axis': bn_axis,
                    'momentum': bn_momentum,
                }
            }
        ])

        if expand_ratio == 1:
            res.append({
                'name': name + 'project_activation',
                'class_name': 'Activation',
                'config': {
                    'activation': activation
                }
            })

        # residual
        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                res.append({
                    'name': name + 'drop',
                    'class_name': 'Dropout',
                    'config': {
                        'rate': survival_probability,
                        'noise_shape': (None, 1, 1, 1, 1),
                    }
                })
            last_layer_name = res[-1]['name']
            res.append({
                'name': name + 'add',
                'class_name': 'Add',
                'inputs': [
                    last_layer_name, inputs
                ]
            })

        return res
    return apply


def EfficientNetV2(
    width_coefficient,
    depth_coefficient,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    min_depth=8,
    bn_momentum=0.9,
    activation="swish",
    blocks_args="default",
    model_name="efficientnetv2",
    classes=1000,
    classifier_activation="softmax",
):
    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS_ARGS[model_name]

    bn_axis = 4 if backend.image_data_format() == "channels_last" else 1

    layers = []

    # Build stem
    stem_filters = round_filters(
        filters=blocks_args[0]["input_filters"],
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor,
    )

    layers.extend([
        {
            'name': 'stem_conv',
            'class_name': 'Conv3D',
            'config': {
                'filters': stem_filters,
                'kernel_size': 3,
                'strides': 2,
                'kernel_initializer': CONV_KERNEL_INITIALIZER,
                'padding': "same",
                'use_bias': False,
            }
        },
        {
            'name': 'stem_bn',
            'class_name': 'BatchNormalization',
            'config': {
                'axis': bn_axis,
                'momentum': bn_momentum
            }
        },
        {
            'name': 'stem_activation',
            'class_name': 'Activation',
            'config': {
                'activation': activation
            }
        }
    ])

    # build block
    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum(args["num_repeat"] for args in blocks_args))

    for (i, args) in enumerate(blocks_args):
        assert args["num_repeat"] > 0

        # Update block input and output filters based on depth multiplier.
        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor)
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor)

        # Determine which conv type to use:
        block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient)

        input_name = layers[-1]['name']
        for j in range(repeats):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            output_layers = block(
                activation=activation,
                bn_momentum=bn_momentum,
                survival_probability=drop_connect_rate * b / blocks,
                name="block{}{}_".format(i + 1, chr(j + 97)),
                **args,
            )(input_name)

            input_name = output_layers[-1]['name']
            layers.extend(output_layers)

            b += 1

    # build top
    top_filters = round_filters(
        filters=1280,
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor)

    layers.extend([
        {
            'name': 'top_conv',
            'class_name': 'Conv3D',
            'config': {
                'filters': top_filters,
                'kernel_size': 1,
                'strides': 1,
                'kernel_initializer': CONV_KERNEL_INITIALIZER,
                'padding': "same",
                'use_bias': False,
            }
        },
        {
            'name': 'top_bn',
            'class_name': 'BatchNormalization',
            'config': {
                'axis': bn_axis,
                'momentum': bn_momentum
            }
        },
        {
            'name': 'top_activation',
            'class_name': 'Activation',
            'config': {
                'activation': activation
            }
        },
        {
            'name': 'avg_pool',
            'class_name': 'GlobalAveragePooling3D'
        }
    ])

    if dropout_rate > 0:
        layers.append(
            {
                'name': 'top_dropout',
                'class_name': 'Dropout',
                'config': {
                    'rate': dropout_rate
                }
            }
        )

    layers.append(
        {
            'name': 'predictions',
            'class_name': 'Dense',
            'config': {
                'units': classes,
                'activation': classifier_activation,
                'kernel_initializer': DENSE_KERNEL_INITIALIZER,
                'bias_initializer': 'Constant',
            }
        }
    )

    return layers


def EfficientNetV2B0(
    classes=1000,
    classifier_activation="softmax",
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-b0",
        classes=classes,
        classifier_activation=classifier_activation)


def EfficientNetV2B1(
    classes=1000,
    classifier_activation="softmax",
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.1,
        model_name="efficientnetv2-b1",
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2B2(
    classes=1000,
    classifier_activation="softmax",
):
    return EfficientNetV2(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        model_name="efficientnetv2-b2",
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2B3(
    classes=1000,
    classifier_activation="softmax",
):
    return EfficientNetV2(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        model_name="efficientnetv2-b3",
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2S(
    classes=1000,
    classifier_activation="softmax",
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-s",
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2M(
    classes=1000,
    classifier_activation="softmax",
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-m",
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2L(
    classes=1000,
    classifier_activation="softmax",
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-l",
        classes=classes,
        classifier_activation=classifier_activation,
    )


def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the EfficientNetV2 model
    implementation. Users are no longer required to call this method to normalize
    the input data. This method does nothing and only kept as a placeholder to
    align the API surface between old and new version of model.

    Args:
      x: A floating point `numpy.array` or a `tf.Tensor`.
      data_format: Optional data format of the image tensor/array. Defaults to
        None, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it, it
        defaults to "channels_last").{mode}

    Returns:
      Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x


def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)

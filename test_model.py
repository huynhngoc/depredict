from deoxys.loaders import load_architecture
import utils.new_layers as new_layers
import json
from utils.build_efficientnet3d_json import EfficientNetV2, EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2S, EfficientNetV2M

layers = EfficientNetV2B0(classes=1, classifier_activation='sigmoid')

with open('effb0.json', 'w') as f:
    json.dump(layers, f)


layers = EfficientNetV2B1(classes=1, classifier_activation='sigmoid')

with open('effb1.json', 'w') as f:
    json.dump(layers, f)

layers = EfficientNetV2B2(classes=1, classifier_activation='sigmoid')

with open('effb2.json', 'w') as f:
    json.dump(layers, f)


layers = EfficientNetV2S(classes=1, classifier_activation='sigmoid')

with open('eff_s.json', 'w') as f:
    json.dump(layers, f)

layers = EfficientNetV2M(classes=1, classifier_activation='sigmoid')

with open('eff_m.json', 'w') as f:
    json.dump(layers, f)


model = load_architecture({
    'type': 'ResNet',
    'layers': layers
}, input_params={
    'shape': (128, 128, 128, 15)
})
model.summary()

# with open('my.json', 'w') as f:
#     json.dump(model.get_config(), f)


block_args = [
    {
        "kernel_size": 3,
        "num_repeat": 1,
        "input_filters": 16,
        "output_filters": 8,
        "expand_ratio": 1,
        "se_ratio": 0,
        "strides": 1,
        "conv_type": 1,
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 8,
        "output_filters": 16,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": 1,
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 16,
        "output_filters": 24,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": 1,
    },
    {
        "kernel_size": 3,
        "num_repeat": 3,
        "input_filters": 24,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": 0,
    },
    {
        "kernel_size": 3,
        "num_repeat": 5,
        "input_filters": 48,
        "output_filters": 56,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": 0,
    },
    {
        "kernel_size": 3,
        "num_repeat": 8,
        "input_filters": 56,
        "output_filters": 96,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": 0,
    },
]

layers = EfficientNetV2(
    width_coefficient=1.0,
    depth_coefficient=1.0,
    # model_name="efficientnetv2-b0",
    classes=1,
    classifier_activation='sigmoid',
    blocks_args=block_args
)


with open('architectures/effb0_half.json', 'w') as f:
    json.dump(layers, f)


model = load_architecture({
    'type': 'ResNet',
    'layers': layers
}, input_params={
    'shape': (173, 191, 265, 2)
})
model.summary()

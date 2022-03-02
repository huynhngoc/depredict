from deoxys.loaders import load_architecture
import utils.new_layers as new_layers
import json
from utils.build_efficientnet3d_json import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2S, EfficientNetV2M

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

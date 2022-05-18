import re
import torch.nn as nn

from torch.utils import model_zoo

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

class ResNetEncoder(ResNet):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels

        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, self.conv1.out_channels, self.conv1.kernel_size, self.conv1.stride, self.conv1.padding, bias=self.conv1.bias)

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def get_out_channels(self):
        return list(self._out_channels)

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)

encoders_param = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "params": {
            "in_channels": 1,
            "out_channels": (64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2]
        }
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "params": {
            "in_channels": 1,
            "out_channels": (256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3]
        }
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
        "params": {
            "in_channels": 1,
            "out_channels": (256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3]
        }
    }
}

def get_encoder(encoder_name, pretrained=True):
    if encoder_name not in encoders_param:
        raise Exception("Wrong encoder name")
    
    Encoder = encoders_param[encoder_name]["encoder"]
    params = encoders_param[encoder_name]["params"]
    encoder = Encoder(**params)

    if pretrained:
        print("Loading ImageNet pretrained weights")
        state_dict = model_zoo.load_url(encoders_param[encoder_name]["url"])
        if params["in_channels"] < 3:
            state_dict['conv1.weight'] = state_dict['conv1.weight'][:, :params["in_channels"], :, :]

        encoder.load_state_dict(state_dict)
    
    return encoder

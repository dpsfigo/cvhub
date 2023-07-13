'''
Author: dpsfigo
Date: 2023-07-13 15:42:44
LastEditors: dpsfigo
LastEditTime: 2023-07-13 19:42:52
Description: 请填写简介
'''
import torch
import torch.nn as nn

cfgs = {
    'vgg11': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    'vgg13':
    [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    'vgg16': [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
        512, 512, 512, "M"
    ],
    'vgg19': [
        64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512,
        512, "M", 512, 512, 512, 512, "M"
    ]
}



class VGG(nn.Module):

    def __init__(self,
                 features: nn.Module,
                 num_classes: int = 1000,
                 dropout: float = 0.5) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_features(cfg:list, batch_norm:bool=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v=="M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,  nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



def vgg(model_name = "vgg16", **kwargs):
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)
    return model
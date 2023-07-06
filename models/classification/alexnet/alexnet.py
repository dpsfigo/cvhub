'''
Author: dpsfigo
Date: 2023-06-27 15:40:01
LastEditors: dpsfigo
LastEditTime: 2023-07-05 14:28:37
Description: AlexNet backbone
'''
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__ (self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x.view(-1,3,224,224)
        x = self.features(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), 256*6*6)
            x = self.classifier(x)
        return x
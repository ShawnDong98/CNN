import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRN(nn.Module):
    def __init__(self, n=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(n, 1, 1),
            stride = 1,
            padding = (int((n / 2)), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=n,
            stride = 1,
            padding = int(n / 2))

        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)

        x = x.div(div)
        return x


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG_FCN(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_FCN, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(4096, num_classes)
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'LRN':
            print("I'm in 1")
            layers += [LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)]
        elif v == 'M':
            print("I'm in 2")
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'Conv1x1_256':
            print("I'm in 3")
            conv2d = nn.Conv2d(256, 256, kernel_size=1, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        elif v == 'Conv1x1_512':
            print("I'm in 4")
            conv2d = nn.Conv2d(512, 512, kernel_size=1, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        else:
            print("I'm in 5")
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        
    return nn.Sequential(*layers)


# configs
cfgs = {
    # 11
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'A-LRN': [64, 'LRN', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 'Conv1x1_256', 'M', 512, 512, 'Conv1x1_512', 'M', 512, 512, 'Conv1x1_512', 'M'], 

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

}


def vgg11(batch_norm=False, num_classes=1000):
    model = VGG(make_layers(cfgs['A']), num_classes=num_classes)
    return model

def vgg11_LRN(batch_norm=False, num_classes=1000):
    model = VGG(make_layers(cfgs['A-LRN']), num_classes=num_classes)
    return model



def vgg13(batch_norm=False, num_classes=1000):
    model = VGG(make_layers(cfgs['B']), num_classes=num_classes)
    return model

def vgg16_Conv1x1(batch_norm=False, num_classes=1000):
    model = VGG(make_layers(cfgs['C']), num_classes=num_classes)
    return model


def vgg16(batch_norm=False, num_classes=1000):
    model = VGG(make_layers(cfgs['D']), num_classes=num_classes)
    return model


def vgg16_FCN(batch_norm=False, num_classes=1000):
    model = VGG_FCN(make_layers(cfgs['D']), num_classes=num_classes)
    return model


def vgg19(batch_norm=False, num_classes=1000):
    model = VGG(make_layers(cfgs['E']), num_classes=num_classes)
    return model


def get_vgg(model, num_classes=1000):
    if model == 'vgg11':
        return vgg11(num_classes=num_classes)

    if model == 'vgg11_LRN':
        return vgg11_LRN(num_classes=num_classes)

    if model == 'vgg13':
        return vgg13(num_classes=num_classes)

    if model == 'vgg16_Conv1x1':
        return vgg16_Conv1x1(num_classes=num_classes)
    
    if model == 'vgg16':
        return vgg16(num_classes=num_classes)

    if model == 'vgg19':
        return vgg19(num_classes=num_classes)
    
    if model == 'vgg16_FCN':
        return vgg16_FCN(num_classes=num_classes)
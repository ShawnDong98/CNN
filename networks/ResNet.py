import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def ConvBlock(in_planes, out_planes, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PlainBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(PlainBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out


class BasicBlockCIFAR(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, norm_layer=None):
        super(BasicBlockCIFAR, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        self.layer1 = ConvBlock(inplanes, inplanes, stride=1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(planes),
        )
        
        

        self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        

        out = self.layer1(x)

        out = self.layer2(out)

        if self.stride != 1:
            x = self.pool(x)
            batch, ch, h, w = x.size()
            extra_dim = torch.zeros((batch, ch, h, w)).to(x.device)
            identity = torch.cat([x, extra_dim], 1)
        else:
            identity = x

        

        out += identity
        out = self.relu(out)

        return out
        

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups 

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, nn.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # blocks -> layers
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)



class ResNet_CIFAR(nn.Module):
    def __init__(self, n, num_classes=1000, shortcut=True):
        super(ResNet_CIFAR, self).__init__()
        self.shortcut = shortcut
        self.num_classes = num_classes
        self.first_layer = ConvBlock(3, 16, stride=1)
        self.layer1 = self._make_layer(16, 32, n)
        self.layer2 = self._make_layer(32, 64, n)
        self.layer3 = self._make_layer(64, 64, n)

        self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1*1*64, self.num_classes)

        
    def _make_layer(self, in_planes, out_planes, n):
        layers = []
        if self.shortcut:
            if in_planes != out_planes:
                for i in range(n-1):
                    layers.append(BasicBlockCIFAR(in_planes, in_planes, stride=1))

                layers.append(BasicBlockCIFAR(in_planes, out_planes, stride=2))
            else:
                for i in range(n-1):
                    layers.append(BasicBlockCIFAR(in_planes, in_planes, stride=1))

                layers.append(BasicBlockCIFAR(in_planes, out_planes, stride=1))
        else:
            if in_planes != out_planes:
                for i in range(2*n-1):
                    layers.append(ConvBlock(in_planes, in_planes, stride=1))

                layers.append(ConvBlock(in_planes, out_planes, stride=2))
            else:
                for i in range(2*n-1):
                    layers.append(ConvBlock(in_planes, in_planes))

                layers.append(ConvBlock(in_planes, out_planes, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out



def resnet18(num_classes=1000):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model

def resnet18_plain(num_classes=1000):
    model = ResNet(PlainBasicBlock, [2, 2, 2, 2], num_classes)
    return model


def resnet34(num_classes=1000):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    return model

def resnet34_plain(num_classes=1000):
    model = ResNet(PlainBasicBlock, [3, 4, 6, 3], num_classes)
    return model

def resnet50(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model


def resnet101(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


def resnet152(num_classes=1000):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
    return model


def plain_net_20(num_classes=1000):
    model = ResNet_CIFAR(3, num_classes=num_classes, shortcut=False)
    return model

def resnet_CIFAR_20(num_classes=1000):
    model = ResNet_CIFAR(3, num_classes=num_classes, shortcut=True)
    return model


def plain_net_32(num_classes=1000):
    model = ResNet_CIFAR(5, num_classes=num_classes, shortcut=False)
    return model

def resnet_CIFAR_32(num_classes=1000):
    model = ResNet_CIFAR(5, num_classes=num_classes, shortcut=True)
    return model


def plain_net_44(num_classes=1000):
    model = ResNet_CIFAR(7, num_classes=num_classes, shortcut=False)
    return model

def resnet_CIFAR_44(num_classes=1000):
    model = ResNet_CIFAR(7, num_classes=num_classes, shortcut=True)
    return model


def plain_net_56(num_classes=1000):
    model = ResNet_CIFAR(9, num_classes=num_classes, shortcut=False)
    return model

def resnet_CIFAR_56(num_classes=1000):
    model = ResNet_CIFAR(9, num_classes=num_classes, shortcut=True)
    return model


def plain_net_110(num_classes=1000):
    model = ResNet_CIFAR(18, num_classes=num_classes, shortcut=False)
    return model

def resnet_CIFAR_110(num_classes=1000):
    model = ResNet_CIFAR(18, num_classes=num_classes, shortcut=True)
    return model

def plain_net_1202(num_classes=1000):
    model = ResNet_CIFAR(200, num_classes=num_classes, shortcut=False)
    return model

def resnet_CIFAR_1202(num_classes=1000):
    model = ResNet_CIFAR(200, num_classes=num_classes, shortcut=True)
    return model



def get_model(model_name, num_classes = 1000):
    if model_name == "resnet18":
        return resnet18(num_classes)    

    if model_name == "resnet18_plain":
        return resnet18_plain(num_classes)
    
    if model_name == "resnet34":
        return resnet34(num_classes)

    if model_name == "resnet34_plain":
        return resnet34_plain(num_classes)

    if model_name == "resnet50":
        return resnet50(num_classes)

    if model_name == "resnet101":
        return resnet101(num_classes)

    if model_name == "resnet152":
        return resnet152(num_classes)


    if model_name == "resnet_CIFAR_20":
        return resnet_CIFAR_20(num_classes)

    if model_name == "plain_net_20":
        return plain_net_20(num_classes)

    if model_name == "resnet_CIFAR_32":
        return resnet_CIFAR_32(num_classes)

    if model_name == "plain_net_32":
        return plain_net_32(num_classes)

    if model_name == "resnet_CIFAR_44":
        return resnet_CIFAR_44(num_classes)

    if model_name == "plain_net_44":
        return plain_net_44(num_classes)


    if model_name == "resnet_CIFAR_56":
        return resnet_CIFAR_56(num_classes)

    if model_name == "plain_net_56":
        return plain_net_56(num_classes)


    if model_name == "resnet_CIFAR_110":
        return resnet_CIFAR_110(num_classes)

    if model_name == "plain_net_110":
        return plain_net_110(num_classes)

    
    if model_name == "resnet_CIFAR_1202":
        return resnet_CIFAR_1202(num_classes)

    if model_name == "plain_net_1202":
        return plain_net_1202(num_classes)


if __name__ == "__main__":
    model = get_model("resnet_CIFAR_20", num_classes=5)
    print(model)
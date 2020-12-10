import torch
import torch.nn as nn
import torch.nn.functional as F

import math


# Block with stage means that the block is a down-sample block, where hierarchical connections are removed. There are only 3 blocks in a net that is stage block.

# depreacated 当Bottle2neck_w 的 base_width = 16 时， 等价于Bottle2neck 
class Bottle2neck(nn.Module):
    expansion = 4 
    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4, stype='normal'):
        """
        inplanes： 前面 1x1 卷积的输入的通道数
        planes： bottleneck 3x3 卷积的输入输出通道数
        后面1x1卷积的输出通道数： planes * self.expansion
        

        n: planes
        s: scale
        w: w * s = n, width = planes / scale

        """
        super(Bottle2neck, self).__init__()

        
        width = int(math.floor(planes / scale))
        # print(width)

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding = 1)

        convs = []
        bns = []
        for i in range(scale-1):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # sp 指 split， 将out在第一个维度按照width大小拆分
        spx = torch.split(out, self.width, 1)

        # 如果scale=4， 那么i = 0, 1, 2
        for i in range(self.scale-1):
            # 如果是stage模式， scale的connection就没了
            if i==0 or self.stype == 'stage':
                sp = spx[i]
                # print(sp.shape)
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))

            if i ==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype=='normal':
            out = torch.cat((out, spx[self.scale-1]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            # b = self.pool(spx[self.scale-1])
            # print(spx[self.scale-1].shape)
            # print("b_shape: ", b.shape)
            out = torch.cat((out, self.pool(spx[self.scale-1])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottle2neck_w(nn.Module):
    expansion = 4 
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width = 26, scale=4, stype='normal'):
        """
        inplanes： 前面 1x1 卷积的输入的通道数
        planes： bottleneck 3x3 卷积的输入输出通道数
        后面1x1卷积的输出通道数： planes * self.expansion
        

        n: planes
        s: scale
        w: w * s = n, width = planes / scale


        """
        super(Bottle2neck_w, self).__init__()

        
        # As long as the model structure and the channel number of each split is the same, there is no difference between different code style, and there should be no difference in performance.
        # Width/baseWidth is just used to control the channel number in each split. We just follow the previous works such as Res2NeXt to use this code style.
        width = int(math.floor((planes * base_width/64)))
        # print(width)

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding = 1)

        convs = []
        bns = []
        for i in range(scale-1):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # sp 指 split， 将out在第一个维度按照width大小拆分
        spx = torch.split(out, self.width, 1)

        # 如果scale=4， 那么i = 0, 1, 2
        for i in range(self.scale-1):
            # 如果是stage模式， scale的connection就没了
            if i==0 or self.stype == 'stage':
                sp = spx[i]
                # print(sp.shape)
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))

            if i ==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype=='normal':
            out = torch.cat((out, spx[self.scale-1]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            # b = self.pool(spx[self.scale-1])
            # print(spx[self.scale-1].shape)
            # print("b_shape: ", b.shape)
            out = torch.cat((out, self.pool(spx[self.scale-1])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers,  base_width = 26, scale = 4, num_classes = 1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.base_width = base_width
        self.scale = scale 

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # blocks -> layers
    # stride 只在 residual 的 downsample 1x1 卷积 和 bottleneck 的 3x3 卷积中起作用
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, stype='stage', base_width = self.base_width, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width = self.base_width,scale=self.scale))

    
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def res2net50(num_classes=1000):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], scale = 4, num_classes = num_classes)
    return model



def res2net50_w(num_classes=1000):
    model = Res2Net(Bottle2neck_w, [3, 4, 6, 3], scale = 4, num_classes = num_classes)
    return model


def res2net50_26w_6s(num_classes=1000):
    model = Res2Net(Bottle2neck_w, [3, 4, 6, 3], base_width = 26, scale = 6, num_classes = num_classes)
    return model


def res2net50_26w_8s(num_classes=1000):
    model = Res2Net(Bottle2neck_w, [3, 4, 6, 3], base_width = 26, scale = 8, num_classes = num_classes)
    return model


def res2net50_48w_2s(num_classes=1000):
    model = Res2Net(Bottle2neck_w, [3, 4, 6, 3], base_width = 48, scale = 2, num_classes = num_classes)
    return model


def res2net50_26w_4s(num_classes=1000):
    model = Res2Net(Bottle2neck_w, [3, 4, 6, 3], base_width = 26, scale = 4, num_classes = num_classes)
    return model


def res2net50_14w_8s(num_classes=1000):
    model = Res2Net(Bottle2neck_w, [3, 4, 6, 3], base_width = 14, scale = 8, num_classes = num_classes)
    return model


def res2net50_16w_4s(num_classes=1000):
    model = Res2Net(Bottle2neck_w, [3, 4, 6, 3], base_width = 16, scale = 4, num_classes = num_classes)
    return model





def get_model(model_name, num_classes=1000):
    if model_name == "res2net50":
        return res2net50(num_classes)

    if model_name == "res2net50_w":
        return res2net50_w(num_classes)

    if model_name == "res2net50_26w_6s":
        return res2net50_26w_6s(num_classes)

    if model_name == "res2net50_26w_6s_dev":
        return res2net50_26w_6s(num_classes)

    if model_name == "res2net50_26w_8s":
        return res2net50_26w_8s(num_classes)

    if model_name == "res2net50_26w_8s_dev":
        return res2net50_26w_8s(num_classes)

    if model_name == "res2net50_48w_2s":
        return res2net50_48w_2s(num_classes)
    
    if model_name == "res2net50_26w_4s":
        return res2net50_26w_4s(num_classes)

    if model_name == "res2net50_14w_8s":
        return res2net50_14w_8s(num_classes)



if __name__ == "__main__":
    downsample = nn.Sequential(
                nn.Conv2d(64, 128,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128),
            )
    block = Bottle2neck_w(64, 64, stride=1, downsample=downsample, scale=4, stype='stage')
    a = torch.ones((1, 64, 32, 32))
    block(a)
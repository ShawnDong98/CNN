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

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)

        return F.relu(x, inplace=True)


class Inception(nn.Module):
    # red指什么？
    # red 指 3x3 和 5x5 卷积前的 1x1 卷积核数量
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]

        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)



class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))

        x = self.conv(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x), inplace=True)

        x = F.dropout(x, 0.7, training=self.training)

        x = self.fc2(x)

        return x
 



class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000,  blocks=None, train=True):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]


        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]


        # 224 x 224 -> 112 x 112 
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        # 112 x 112 -> 56 x 56
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.LRN1 = LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.LRN2 = LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        # 56 x 56 -> 28 x 28
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)


        
        # inchannel, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj
        # output = 64 + 128 + 32 + 32 = 256
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        # output = 128 + 192 + 96 + 64 = 480
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        # 28 x 28 -> 14 x 14
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 192 + 208 + 48 + 64 = 512
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        # 160 + 224 + 64 + 64 = 512
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        # 128 + 256 + 64 + 64 = 512
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        # 112 + 288 + 288 + 64 = 528
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        # 256 + 320 + 128 + 128 = 832
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        # 14 x 14 -> 7 x 7
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 256 + 320 + 128 + 128 = 832
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        # 384 + 384 + 128 + 128 = 1024
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)


        # inception4a 之后
        self.aux1 = inception_aux_block(512, num_classes)
        # inception4d 之后
        self.aux2 = inception_aux_block(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)


        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training:
            return x, aux1, aux2
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class GoogLeNet_WithoutAux(nn.Module):

    def __init__(self, num_classes=1000, blocks=None):
        super(GoogLeNet_WithoutAux, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]


        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]


        # 224 x 224 -> 112 x 112 
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        # 112 x 112 -> 56 x 56
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.LRN1 = LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.LRN2 = LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        # 56 x 56 -> 28 x 28
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)


        
        # inchannel, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj
        # output = 64 + 128 + 32 + 32 = 256
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        # output = 128 + 192 + 96 + 64 = 480
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        # 28 x 28 -> 14 x 14
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 192 + 208 + 48 + 64 = 512
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        # 160 + 224 + 64 + 64 = 512
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        # 128 + 256 + 64 + 64 = 512
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        # 112 + 288 + 288 + 64 = 528
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        # 256 + 320 + 128 + 128 = 832
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        # 14 x 14 -> 7 x 7
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 256 + 320 + 128 + 128 = 832
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        # 384 + 384 + 128 + 128 = 1024
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)


        x = self.maxpool3(x)
        x = self.inception4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x



if __name__ == "__main__":
    x = None
    a = 1
    b = x(a)
    print(b)
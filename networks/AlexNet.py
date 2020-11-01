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


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.Conv_layer1 = nn.Sequential(
            # 224 -> 55
            nn.Conv2d(3, 96, 11, stride=4, padding=2),  
            LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True),
            nn.ReLU(),
            # 55 -> 27
            nn.MaxPool2d(3, 2)   
        )
        self.Conv_layer2 = nn.Sequential(
            # 27 -> 27
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True),
            nn.ReLU(),
            # 27 -> 13
            nn.MaxPool2d(3, 2)
        )
        self.Conv_layer3 = nn.Sequential(
            # 13 -> 13
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.Conv_layer4 = nn.Sequential(
            # 13 -> 13
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.Conv_layer5 = nn.Sequential(
            # 13 -> 13
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            # 13 -> 6
            nn.MaxPool2d(3, 2),
        )

        self.FC_layer1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(), 
        ) 
        self.FC_layer2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.FC_layer3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.Conv_layer1(x)
        out = self.Conv_layer2(out)
        out = self.Conv_layer3(out)
        out = self.Conv_layer4(out)
        out = self.Conv_layer5(out)

        out = out.view(out.size(0), -1)
        out = self.FC_layer1(out)
        out = self.FC_layer2(out)
        out = self.FC_layer3(out)

        return out


class AlexNet_pytorch(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_pytorch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.Conv_layer1 = nn.Sequential(
            # 224 -> 55
            nn.Conv2d(3, 48, 11, stride=4, padding=2),  
            LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True),
            nn.ReLU(),
            # 55 -> 27
            nn.MaxPool2d(3, 2)   
        )
        self.Conv_layer2 = nn.Sequential(
            # 27 -> 27
            nn.Conv2d(48, 128, 5, stride=1, padding=2),
            LRN(5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True),
            nn.ReLU(),
            # 27 -> 13
            nn.MaxPool2d(3, 2)
        )

    def forward(self, x):
        out = self.Conv_layer1(x)
        out = self.Conv_layer2(out)

        return out


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.Conv_layer3 = nn.Sequential(
            # 13 -> 13
            nn.Conv2d(128, 192, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.Conv_layer4 = nn.Sequential(
            # 13 -> 13
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.Conv_layer5 = nn.Sequential(
            # 13 -> 13
            nn.Conv2d(192, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            # 13 -> 6
            nn.MaxPool2d(3, 2),
        )

    def forward(self, x):
        out = self.Conv_layer3(x)
        out = self.Conv_layer4(out)
        out = self.Conv_layer5(out)

        return out


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.FC_layer1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(),   
        ) 

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.FC_layer1(out)

        return out


class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.FC_layer2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),  
        )
    
    def forward(self, x):
        out = self.FC_layer2(x)

        return out

class Model5(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model5, self).__init__()
        self.FC_layer3 = nn.Linear(4096, num_classes)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.FC_layer3(out)

        return out


class AlexNet_Split(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_Split, self).__init__()
        self.Model11 = Model1()
        self.Model12 = Model1()

        self.Model21 = Model2()
        self.Model22 = Model2()

        self.Model31 = Model3()
        self.Model32 = Model3()

        self.Model41 = Model4()
        self.Model42 = Model4()

        self.Model5 = Model5(num_classes)

    def forward(self, x, flag):
        if flag == 0:
            out11 = self.Model11(x)
            out12 = self.Model12(x)

            out21 = self.Model21(out11)
            out22 = self.Model22(out12)

            out31 = self.Model31(out21)
            out32 = self.Model32(out22)

            out41 = self.Model41(out31)
            out42 = self.Model42(out32)

            out5 = self.Model5(out41, out42)

            return out5

        if flag == 1:
            out11 = self.Model11(x)
            out12 = self.Model12(x)

            out21 = self.Model21(out12)
            out22 = self.Model22(out11)

            out31 = self.Model31(out21)
            out32 = self.Model32(out22)

            out41 = self.Model41(out31)
            out42 = self.Model42(out32)

            out5 = self.Model5(out41, out42)

            return out5

        if flag == 2:
            out11 = self.Model11(x)
            out12 = self.Model12(x)

            out21 = self.Model21(out11)
            out22 = self.Model22(out12)

            out31 = self.Model31(out22)
            out32 = self.Model32(out21)

            out41 = self.Model41(out31)
            out42 = self.Model42(out32)

            out5 = self.Model5(out41, out42)

            return out5

        if flag == 3:
            out11 = self.Model11(x)
            out12 = self.Model12(x)

            out21 = self.Model21(out12)
            out22 = self.Model22(out11)

            out31 = self.Model31(out22)
            out32 = self.Model32(out21)

            out41 = self.Model41(out31)
            out42 = self.Model42(out32)

            out5 = self.Model5(out41, out42)

            return out5

        if flag == 4:
            out11 = self.Model11(x)
            out12 = self.Model12(x)

            out21 = self.Model21(out11)
            out22 = self.Model22(out12)

            out31 = self.Model31(out21)
            out32 = self.Model32(out22)

            out41 = self.Model41(out32)
            out42 = self.Model42(out31)

            out5 = self.Model5(out41, out42)

            return out5

        if flag == 5:
            out11 = self.Model11(x)
            out12 = self.Model12(x)

            out21 = self.Model21(out12)
            out22 = self.Model22(out11)

            out31 = self.Model31(out21)
            out32 = self.Model32(out22)

            out41 = self.Model41(out32)
            out42 = self.Model42(out31)

            out5 = self.Model5(out41, out42)

            return out5

        if flag == 6:
            out11 = self.Model11(x)
            out12 = self.Model12(x)

            out21 = self.Model21(out11)
            out22 = self.Model22(out12)

            out31 = self.Model31(out22)
            out32 = self.Model32(out21)

            out41 = self.Model41(out32)
            out42 = self.Model42(out31)

            out5 = self.Model5(out41, out42)

            return out5


        if flag == 7:
            out11 = self.Model11(x)
            out12 = self.Model12(x)

            out21 = self.Model21(out12)
            out22 = self.Model22(out11)

            out31 = self.Model31(out22)
            out32 = self.Model32(out21)

            out41 = self.Model41(out32)
            out42 = self.Model42(out31)

            out5 = self.Model5(out41, out42)

            return out5

        





    









if __name__ == "__main__":
    n = 5
    print(int(n/2))
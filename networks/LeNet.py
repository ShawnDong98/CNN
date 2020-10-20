import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        # MNIST
        # self.conv_1 = nn.Conv2d(1, 6, 5)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        # CIFAR10
        self.fc_1 = nn.Linear(400, 120)
        # MNIST
        # self.fc_1 = nn.Linear(256, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)


    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


class LeNet_Linear(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet_Linear, self).__init__()
        # self.conv_1 = nn.Conv2d(1, 6, 5)
        # self.conv_2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(32 * 32 * 3, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.fc_1 = nn.Linear(256, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)


    def forward(self, x):
        out = x.view(x.size(0), -1)
        # print(out.shape)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out

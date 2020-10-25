import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from torchvision import datasets
from torchvision import transforms


import numpy as np
import matplotlib.pyplot as plt

from networks.LeNet import FeatureLeNet
from util import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class get_features():
    def __init__(self):
        super(get_features, self).__init__()
        self.trans = transforms.ToTensor()
        self.testset = datasets.CIFAR10(
            root = "../datasets/",
            train=False,
            download=True,
            transform=self.trans
        )

        self.test_loader = DataLoader(
            self.testset,
            batch_size=1,
            shuffle=True
        )


        self.net_init()


    def net_init(self):
        self.net = FeatureLeNet().to(device)
        self.net.load_state_dict(torch.load("../saved_models/LeNet/bs20_lr004_FeatureLeNet_CIFAR10.pth"))
        self.net.eval()

    def viz(self, module, input, output):
        # print(output.shape)
        x = output[0].cpu()
        # print(x.shape)
        # 最多显示4张图
        min_num = np.minimum(16, x.size()[0])
        for i in range(min_num):
            plt.subplot(4, 4, i+1)
            plt.imshow(x[i])
        plt.savefig("../plots/LeNet/features/"+str(min_num)+".jpg")
        plt.show()

    def run(self):
        for name, m in self.net.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                # print(name)
                # print(m)
                m.register_forward_hook(self.viz)

        loader = iter(self.test_loader)

        data, label = next(loader)
        data = data.to(device)

        print(data.shape)
        print(label)

        img = tensor_to_PIL(data)
        print(type(img))
        plt.imshow(img)
        plt.savefig('src.jpg')
        plt.show()
        with torch.no_grad():
            self.net(data)


if __name__ == "__main__":
    g = get_features()
    g.run()


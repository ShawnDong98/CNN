import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  

unloader = transforms.ToPILImage()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

# 直接展示tensor格式图片
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def concat2pic(img_path1, img_path2):
    img1 = cv2.imread("../plots/LeNet/losses/Linear.jpg")
    img2 = cv2.imread("../plots/LeNet/losses/LeNet.jpg")

    print(img1.shape)
    print(img2.shape)

    lr04 = np.concatenate([img1, img2], axis=0)

    # cv2.imshow("merge", merge)
    cv2.imwrite("../plots/LeNet/losses/lr04.jpg", lr04)




if __name__ == "__main__":
    concat2pic()
        
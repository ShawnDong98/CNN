import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

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


def concat2pic(img_path1, img_path2, saved_path):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    print(img1.shape)
    print(img2.shape)

    lr04 = np.concatenate([img1, img2], axis=1)

    # cv2.imshow("merge", merge)
    cv2.imwrite(saved_path, lr04)


def compute_complexity_params(model):
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))



if __name__ == "__main__":
    img_path1 = "../plots/DenseNet/losses/bs64_lr1_densenet_L100_k12.jpg"
    img_path2 = "../plots/DenseNet/losses/bs64_lr1_resnet_CIFAR_110.jpg"
    saved_path = "../plots/DenseNet/losses/DenseNet_ResNet_compare.jpg"
    concat2pic(img_path1, img_path2, saved_path)
        
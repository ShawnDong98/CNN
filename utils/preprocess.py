import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import cv2
import numpy as np
from PIL import Image
import random

toPIL = transforms.ToPILImage()


def random_crop_and_flip(img, src_size, dst_size, train=True):
    
    img = img.resize((src_size, src_size))
    # img.save("../plots/VGGNet/features/256src.jpg")
    img = np.array(img)

    x_offset = np.random.randint(0, src_size-dst_size+1)
    y_offset = np.random.randint(0, src_size-dst_size+1)
    # print(x_offset, y_offset)

    crop = img[x_offset:x_offset+dst_size, y_offset:y_offset+dst_size, :]
    
    if train:
        prob = np.random.randint(0, 2)
        # print(prob)
        if prob == 1:
            crop = np.flip(crop, 1)

    crop = toPIL(crop)
    # crop.show()
    # crop.save("../plots/VGGNet/features/256crop.jpg")
    return crop


def random_scale_jitter_and_flip(img, dst_size):
    # [256, 513)
    src_size = np.random.randint(256, 513)
    # img.save("../plots/VGGNet/features/" + str(src_size) + "src.jpg")
    # print(src_size)

    img = img.resize((src_size, src_size))
    img = np.array(img)

    x_offset = np.random.randint(0, src_size-dst_size+1)
    y_offset = np.random.randint(0, src_size-dst_size+1)

    crop = img[x_offset:x_offset+dst_size, y_offset:y_offset+dst_size, :]
    
    prob = np.random.randint(0, 2)
    # print(prob)
    if prob == 1:
        crop = np.flip(crop, 1)

    crop = toPIL(crop)
    # crop.show()
    # crop.save("../plots/VGGNet/features/" + str(src_size) + "crop.jpg")
    return crop


def multi_scale(img, img_size):
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    scales = [img_size-32, img_size, img_size+32]
    imgs = []
    img_copy = img
    for scale in scales:
        img = random_crop_and_flip(img_copy, scale, 224, train=False)
        # img.show()
        img = trans(img)
        imgs.append(img)

    return imgs





def multi_scale_jitter(img, low_size, high_size):
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    scales = [low_size, int((low_size+high_size)/2), high_size]
    imgs = []
    img_copy = img
    for scale in scales:
        img = random_crop_and_flip(img_copy, scale, 224, train=False)
        # img.show()
        img = trans(img)
        imgs.append(img)

    return imgs



def multi_crop(img):
    trans = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    scales = [256, 384, 512]
    crops = []
    img_copy = img
    for scale in scales:
        img = img_copy.resize((scale, scale))
        img = np.array(img)
        img_flip = np.flip(img, 1)
        crop_size = int(scale / 5)
        for j in range(5):
            for k in range(5):
                crop = img[j*crop_size:(j+1)*crop_size, k*crop_size:(k+1)*crop_size]
                crop = toPIL(crop)
                crop = trans(crop)
                crops.append(crop)
                crop_flip = img_flip[j*crop_size:(j+1)*crop_size, k*crop_size:(k+1)*crop_size]
                crop_flip = toPIL(crop_flip)
                crop_flip = trans(crop_flip)
                crops.append(crop_flip)
    # print(len(crops))

    return crops



    










if __name__ == "__main__":
    img = Image.open("/home/shawn/ST/Shawn/CNN/datasets/FlowerImage/train/daisy/5547758_eea9edfd54_n.jpg")

    imgs = multi_scale(img, 256)

    # random_scale_jitter_and_flip(img, 224)
    # multi_crop(img)
    # x_offset = np.random.randint(0, 1)
    # print(x_offset)
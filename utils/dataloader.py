import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
from torchvision import transforms

import os
from shutil import move
import glob
from PIL import Image
from tqdm import tqdm

from preprocess import *



class Dataset(Dataset):
    def __init__(self, root, src_size, dst_size, scale_mode='single_scale', trans=None):
        self.src_size = src_size
        self.dst_size = dst_size
        self.scale_mode = scale_mode
        # 用来存放数据路径和标签
        self.imgs = []
        # 制作index和class的映射
        self.label_map = {}
        i = 0
        # 先把各个类的文件夹找到
        paths = sorted(glob.glob(root + "/*"))
        for path in paths:

            self.label_map[path.split("/")[-1]] = i
            single_class = glob.glob(path + "/*")
            # 读取每一个class的文件夹下的数据
            for img in single_class:
                self.imgs.append((img, i))

            i += 1
        self.trans = trans
        # print(len(self.imgs))
        

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path)
        if self.scale_mode == 'single_scale':
            img = random_crop_and_flip(img, self.src_size, self.dst_size)

        if self.scale_mode == 'scale_jitter':
            # print("I'm in ...")
            img = random_scale_jitter_and_flip(img, self.dst_size)
        if self.scale_mode == 'orgin_scale':
            img = img.resize((self.dst_size, self.dst_size))

        if self.scale_mode == 'multi_crop':
            img = multi_crop(img)

        if self.scale_mode == 'multi_scale':
            img = multi_scale(img, self.src_size)

        if self.scale_mode == 'multi_scale_jitter':
            img = multi_scale_jitter(img, 256, 512)
        
        if self.trans:
            # print(self.trans)
            img = self.trans(img)

        return (img, label)

    def __len__(self):
        return len(self.imgs)

def ImageLoader(root, src_size, dst_size, batch_size, scale_mode='single_scale', train=True, trans=None):
    if train:
        if trans:
            train_trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
            dataset = Dataset(root, src_size, dst_size, scale_mode, train_trans)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
        else:
            dataset = Dataset(root, src_size, dst_size, scale_mode)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
    else:
        if trans:
            test_trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
            dataset = Dataset(root, src_size, dst_size, scale_mode, test_trans)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False
            )
        else:
            dataset = Dataset(root, src_size, dst_size, scale_mode)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False
            )

    return loader, dataset


def move_files():

    root = "../datasets/ImageNet/"
    dirs = os.listdir(root + "train")

    for dir in dirs:
        if not os.path.exists(os.path.join(root + "test/", dir)):
            os.mkdir(os.path.join(root + "test/", dir))
            files = os.listdir(os.path.join(root + "train/", dir))
            for file in files[:10]:
                if not os.path.exists(os.path.join(root + "test/" + dir + "/", file)):
                    move(os.path.join(root + "train/" + dir + "/", file), os.path.join(root + "test/" + dir + "/", file))
                    print(os.path.join(root + "train/" + dir + "/", file))
        else:
            files = os.listdir(os.path.join(root + "train/", dir))
            for file in files[:10]:
                if not os.path.exists(os.path.join(root + "test/" + dir + "/", file)):
                    move(os.path.join(root + "train/" + dir + "/", file), os.path.join(root + "test/" + dir + "/", file))
                    print(os.path.join(root + "train/" + dir + "/", file))





if __name__ == "__main__":
    root = "../datasets/FlowerImage/train"
    # dirs = sorted(glob.glob(root + "test" + "/*"))
    # print(dirs[0].split("/")[-1])

    # print(dirs)

    # trans = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #         ]
    #     )
    # dataset = Dataset(root, trans)
    # loader = iter(DataLoader(dataset),
    # )

    # x, y = next(loader)


    multi_scale_loader, multi_scale_dataset = ImageLoader("../datasets/FlowerImage/test", 256, 224, 1, scale_mode="multi_scale", train=False)

    # loader = iter(loader)
    # x, y = next(loader)

    for data, label in tqdm(loader):
        print(type(label))
        print(len(data))

    # print(len(x))
    # print(x[0].shape)
    # print(y)


    # for dir in dirs:
    #     if(len(os.listdir(os.path.join(root + "test", dir))) != 10):
    #         print(os.path.join(root + "test", dir))
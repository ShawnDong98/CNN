import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets

import argparse
from tqdm import tqdm
import time


from networks.LeNet import LeNet, LeNet_Linear, FeatureLeNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class trainer_LeNet():
    def __init__(self, config):
        self.config = config

        #--------------------------------------
        # 初始化数据集
        #--------------------------------------
        self.trans = transforms.ToTensor()

        # # MNIST
        # self.trainset = datasets.MNIST(
        #     root = config.img_path,
        #     train=True,
        #     download=True,
        #     transform=self.trans
        # )

        # CIFAR10
        self.trainset = datasets.CIFAR10(
            root = config.img_path,
            train=True,
            download=True,
            transform=self.trans
        )

       

        # # MNIST
        # self.testset = datasets.MNIST(
        #     root = config.img_path,
        #     train=False,
        #     download=True,
        #     transform=self.trans
        # )

        # CIFAR10
        self.testset = datasets.CIFAR10(
            root = config.img_path,
            train=False,
            download=True,
            transform=self.trans
        )

        
        self.train_loader = DataLoader(
            self.trainset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=config.batch_size,
            shuffle=False
        )

        self.net_init()


    def net_init(self):
        try:
            self.net = LeNet().to(device)
            self.net.load_state_dict(torch.load("./saved_models/LeNet/bs20_lr004_LeNet_CIFAR10.pth"))
            self.state = torch.load("./saved_models/LeNet/bs20_lr004_LeNet_CIFAR10_State.pth")
            print("model loaded...")

        except:
            print("No pretrained model...")
            self.net = LeNet().to(device)
            self.state = {
                "epoch": 0,
                "total_loss": [],
                "test_accuracy": []
            }
        

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=0.04)

    def train(self):
        pbar = tqdm(range(self.state["epoch"], self.config.epochs))
        for i in pbar:
            self.net.train()
            loss_sum = 0.0
            pbar.set_description(f"Now the epoch is {i}")
            for num, (data, label) in enumerate(self.train_loader):
                # print(data)
                # print(label)
        
                data = data.to(device)
                label = label.to(device)

                output = self.net(data)
                loss = self.criterion(output, label)
                loss_sum += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(f"loss: {loss}")
            self.state["total_loss"].append(loss_sum)
            print(f"epoch: {i}, loss:{loss_sum}")
            accuracy = self.test()
            self.state["test_accuracy"].append(accuracy)
            print(f"test accuracy: {accuracy}%")
            torch.save(self.net.state_dict(), './saved_models/LeNet/bs20_lr004_LeNet_CIFAR10.pth')
            self.state['epoch'] = i
            torch.save(self.state, './saved_models/LeNet/bs20_lr004_LeNet_CIFAR10_State.pth')


    def test(self):
        self.net.eval()
        total = 0
        correct = 0
        for num, (data, label) in enumerate(self.test_loader):
            data = data.to(device)
            # print(data.shape)
            label = label.to(device)
            output = self.net(data)

            # print(output.shape)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum()

        accuracy = 100 * correct / total

        print(f"accuracy: {accuracy}%")

        return accuracy


class trainer_Linear():
    def __init__(self, config):
        self.config = config

        #--------------------------------------
        # 初始化数据集
        #--------------------------------------
        self.trans = transforms.ToTensor()

        # # MNIST
        # self.trainset = datasets.MNIST(
        #     root = config.img_path,
        #     train=True,
        #     download=True,
        #     transform=self.trans
        # )

        # CIFAR10
        self.trainset = datasets.CIFAR10(
            root = config.img_path,
            train=True,
            download=True,
            transform=self.trans
        )

        # # MNIST
        # self.testset = datasets.MNIST(
        #     root = config.img_path,
        #     train=False,
        #     download=True,
        #     transform=self.trans
        # )

        # CIFAR10
        self.testset = datasets.CIFAR10(
            root = config.img_path,
            train=False,
            download=True,
            transform=self.trans
        )

        self.train_loader = DataLoader(
            self.trainset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=config.batch_size,
            shuffle=False
        )

        self.net_init()


    def net_init(self):
        try:
            self.net = LeNet_Linear().to(device)
            self.net.load_state_dict(torch.load("./saved_models/LeNet/bs20_lr004_Linear_CIFAR10.pth"))
            self.state = torch.load("./saved_models/LeNet/bs20_lr004_Linear_CIFAR10_State.pth")
            print("model loaded...")

        except:
            print("No pretrained model...")
            self.net = LeNet_Linear().to(device)
            self.state = {
                "epoch": 0,
                "total_loss": [],
                "test_accuracy": []
            }
        

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=0.04)

    def train(self):
        pbar = tqdm(range(self.state["epoch"], self.config.epochs))
        for i in pbar:
            self.net.train()
            loss_sum = 0.0
            pbar.set_description(f"Now the epoch is {i}")
            for num, (data, label) in enumerate(self.train_loader):
                # print(data.shape)
                # print(label.shape)

                data = data.to(device)
                label = label.to(device)

                output = self.net(data)
                loss = self.criterion(output, label)
                loss_sum += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(f"loss: {loss}")
            self.state["total_loss"].append(loss_sum)
            print(f"epoch: {i}, loss:{loss_sum}")
            accuracy = self.test()
            self.state["test_accuracy"].append(accuracy)
            print(f"test accuracy: {accuracy}%")
            torch.save(self.net.state_dict(), './saved_models/LeNet/bs20_lr004_Linear_CIFAR10.pth')
            self.state['epoch'] = i
            torch.save(self.state, './saved_models/LeNet/bs20_lr004_Linear_CIFAR10_State.pth')


    def test(self):
        self.net.eval()
        total = 0
        correct = 0
        for num, (data, label) in enumerate(self.test_loader):
            data = data.to(device)
            # print(data.shape)
            label = label.to(device)
            output = self.net(data)

            # print(output.shape)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum()

        accuracy = 100 * correct / total

        # print(f"accuracy: {accuracy}%")

        return accuracy


class FeatureTrainer_LeNet():
    def __init__(self, config):
        self.config = config

        #--------------------------------------
        # 初始化数据集
        #--------------------------------------
        self.trans = transforms.ToTensor()

        # # MNIST
        # self.trainset = datasets.MNIST(
        #     root = config.img_path,
        #     train=True,
        #     download=True,
        #     transform=self.trans
        # )

        # CIFAR10
        self.trainset = datasets.CIFAR10(
            root = config.img_path,
            train=True,
            download=True,
            transform=self.trans
        )

       

        # # MNIST
        # self.testset = datasets.MNIST(
        #     root = config.img_path,
        #     train=False,
        #     download=True,
        #     transform=self.trans
        # )

        # CIFAR10
        self.testset = datasets.CIFAR10(
            root = config.img_path,
            train=False,
            download=True,
            transform=self.trans
        )

        
        self.train_loader = DataLoader(
            self.trainset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=config.batch_size,
            shuffle=False
        )

        self.net_init()


    def net_init(self):
        try:
            self.net = FeatureLeNet().to(device)
            self.net.load_state_dict(torch.load("./saved_models/LeNet/bs20_lr004_FeatureLeNet_CIFAR10.pth"))
            self.state = torch.load("./saved_models/LeNet/bs20_lr004_FeatureLeNet_CIFAR10_State.pth")
            print("model loaded...")

        except:
            print("No pretrained model...")
            self.net = FeatureLeNet().to(device)
            self.state = {
                "epoch": 0,
                "total_loss": [],
                "test_accuracy": []
            }
        

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=0.004)

    def train(self):
        pbar = tqdm(range(self.state["epoch"], self.config.epochs))
        for i in pbar:
            self.net.train()
            loss_sum = 0.0
            pbar.set_description(f"Now the epoch is {i}")
            for num, (data, label) in enumerate(self.train_loader):
                # print(data)
                # print(label)
        
                data = data.to(device)
                label = label.to(device)

                output = self.net(data)
                loss = self.criterion(output, label)
                loss_sum += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(f"loss: {loss}")
            self.state["total_loss"].append(loss_sum)
            print(f"epoch: {i}, loss:{loss_sum}")
            accuracy = self.test()
            self.state["test_accuracy"].append(accuracy)
            print(f"test accuracy: {accuracy}%")
            torch.save(self.net.state_dict(), './saved_models/LeNet/bs20_lr004_FeatureLeNet_CIFAR10.pth')
            self.state['epoch'] = i
            torch.save(self.state, './saved_models/LeNet/bs20_lr004_FeatureLeNet_CIFAR10_State.pth')


    def test(self):
        self.net.eval()
        total = 0
        correct = 0
        for num, (data, label) in enumerate(self.test_loader):
            data = data.to(device)
            # print(data.shape)
            label = label.to(device)
            output = self.net(data)

            # print(output.shape)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum()

        accuracy = 100 * correct / total

        print(f"accuracy: {accuracy}%")

        return accuracy





def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    # 没用到
    parser.add_argument('--img_size', type=int, default=256)
    # 没用到
    parser.add_argument('--total_iter', type=int, default=500000)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--img_path', type=str, default='./datasets')



    return parser.parse_args()







if __name__ == "__main__":
    config = get_config()
    print(config.img_path)
    # trainer = trainer_Linear(config)
    # trainer = trainer_LeNet(config)
    trainer = FeatureTrainer_LeNet(config)
    trainer.train() 


    
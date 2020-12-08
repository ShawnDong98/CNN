import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import densenet121
from torchvision import datasets
from torchvision import transforms

import argparse
import os
from tqdm import tqdm
import time

from networks.DenseNet import *
from utils.plot import plot_loss_accuracy
from utils.dataloader import ImageLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trainer(nn.Module):
    def __init__(self, config):
        super(trainer, self).__init__()
        self.config = config
        self.train_trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.RandomRotation((-5, 5)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.test_trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.trainset = datasets.ImageFolder(
            root = os.path.join(config.img_path, "train"),
            transform=self.train_trans
        )

        self.train_loader = DataLoader(
            self.trainset, 
            batch_size=config.batch_size,
            shuffle=True
        )

        self.testset = datasets.ImageFolder(
            root = os.path.join(config.img_path, "test"),
            transform=self.test_trans
        )

        self.test_loader = DataLoader(
            self.testset, 
            batch_size=config.batch_size,
            shuffle=False
        )

        
        self.net_init()


    def net_init(self):
        try:
            self.net = get_model(self.config.model_name, num_classes=5).to(device)
            # self.net = densenet121(num_classes=10).to(device)
            self.model_optim_lr = torch.load("./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".pth")
            self.net.load_state_dict(self.model_optim_lr['model'])

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.learning_rate, momentum=0.9)
            self.optimizer.load_state_dict(self.model_optim_lr['optimizer'])

            self.scheduler = self.model_optim_lr['lr_scheduler']

            self.state = torch.load("./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth")
        

            print("Model loaded...")

        except:
            self.net = get_model(self.config.model_name, num_classes=5).to(device)
            # self.net = densenet121(num_classes=10).to(device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=self.config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 225], gamma=0.1)
            self.state = {
                "epoch": 0,
                "train_loss": [],
                "test_loss": [], 
                "train_accuracy": [], 
                "test_accuracy": []
            }

            self.model_optim_lr = {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.scheduler
            }
            print("No pretrained model...")


        

    def train(self):
        pbar = tqdm(range(self.state["epoch"], self.config.epochs))
        for i in pbar:
            self.net.train()
            train_loss_sum = 0.0
            train_loss = 0
            train_total = 0
            train_correct = 0
            pbar.set_description(f"Now the epoch is {i}")
            start_time = time.time()
            pbar1 = tqdm(self.train_loader)
            for data, label in pbar1:
                data = data.to(device)
                label = label.to(device)

                output = self.net(data)

                loss = self.criterion(output, label)

                # print(loss)
         
                train_loss_sum += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(output, 1)
                train_total += label.size(0)
                train_correct += (predicted == label).sum()


            epoch_time = time.time() - start_time
            # print(train_total)
            train_accuracy = 100 * train_correct / train_total
            train_loss = train_loss_sum / train_total
            self.state["train_loss"].append(train_loss_sum.cpu().detach().numpy())
            self.state["train_accuracy"].append(train_accuracy.cpu().detach().numpy())
            


            test_loss_sum, test_loss, test_accuracy = self.test()
            self.state["test_loss"].append(test_loss_sum.cpu().detach().numpy())
            self.state["test_accuracy"].append(test_accuracy.cpu().detach().numpy())

            self.scheduler.step()
            print(self.optimizer)

            print("epoch_time: {:.4f}s, epoch: {}, train_loss:{}, train_accuracy: {}".format(epoch_time, i, train_loss, train_accuracy))
            print("epoch_time: {:.4f}s, epoch: {}, test_loss:{}, test_accuracy: {}".format(epoch_time, i, test_loss, test_accuracy))

            self.model_optim_lr['model'] = self.net.state_dict()
            self.model_optim_lr['optimizer'] = self.optimizer.state_dict()
            self.model_optim_lr['lr_scheduler'] = self.scheduler
            torch.save(self.model_optim_lr, "./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".pth")

            self.state['epoch'] = i+1
            torch.save(self.state, "./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth")

            # if i % 10 == 0:
            self.plot()
            

    def test(self):
        self.net.eval()
        test_loss_sum = 0
        test_loss_avg = 0
        test_accuracy = 0
        test_total = 0
        test_correct = 0
        
        for num, (data, label) in enumerate(self.test_loader):
            data = data.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = self.net(data)
                test_loss_sum += self.criterion(output, label)

            _, predicted = torch.max(output, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum()
        

        test_accuracy = 100 * test_correct / test_total
        test_loss_avg = test_loss_sum/test_total

        print(f"test accuracy: {test_accuracy}")


        return test_loss_sum, test_loss_avg, test_accuracy

    def plot(self):
        plot_loss_accuracy("./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth", "./plots/DenseNet/losses/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".jpg", self.trainset, self.testset)



class trainer_CIFAR10(nn.Module):
    def __init__(self, config):
        super(trainer_CIFAR10, self).__init__()
        self.config = config
        self.train_trans = transforms.Compose([
                transforms.Resize((36, 36)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32)),
                transforms.RandomRotation((-5, 5)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # CIFAR10
        self.trainset = datasets.CIFAR10(
            root = config.img_path,
            train=True,
            download=True,
            transform=self.train_trans
        )

        # CIFAR10
        self.testset = datasets.CIFAR10(
            root = config.img_path,
            train=False,
            download=True,
            transform=self.test_trans
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
            self.net = get_model(self.config.model_name, num_classes=10).to(device)
            # self.net = densenet121(num_classes=10).to(device)
            self.model_optim_lr = torch.load("./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".pth")
            self.net.load_state_dict(self.model_optim_lr['model'])

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.learning_rate, momentum=0.9)
            self.optimizer.load_state_dict(self.model_optim_lr['optimizer'])

            self.scheduler = self.model_optim_lr['lr_scheduler']

            self.state = torch.load("./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth")
        

            print("Model loaded...")

        except:
            self.net = get_model(self.config.model_name, num_classes=10).to(device)
            # self.net = densenet121(num_classes=10).to(device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=self.config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 225], gamma=0.1)
            self.state = {
                "epoch": 0,
                "train_loss": [],
                "test_loss": [], 
                "train_accuracy": [], 
                "test_accuracy": []
            }

            self.model_optim_lr = {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.scheduler
            }
            print("No pretrained model...")


        

    def train(self):
        pbar = tqdm(range(self.state["epoch"], self.config.epochs))
        for i in pbar:
            self.net.train()
            train_loss_sum = 0.0
            train_loss = 0
            train_total = 0
            train_correct = 0
            pbar.set_description(f"Now the epoch is {i}")
            start_time = time.time()
            pbar1 = tqdm(self.train_loader)
            for data, label in pbar1:
                data = data.to(device)
                label = label.to(device)

                output = self.net(data)

                loss = self.criterion(output, label)

                # print(loss)
         
                train_loss_sum += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(output, 1)
                train_total += label.size(0)
                train_correct += (predicted == label).sum()


            epoch_time = time.time() - start_time
            # print(train_total)
            train_accuracy = 100 * train_correct / train_total
            train_loss = train_loss_sum / train_total
            self.state["train_loss"].append(train_loss_sum.cpu().detach().numpy())
            self.state["train_accuracy"].append(train_accuracy.cpu().detach().numpy())
            


            test_loss_sum, test_loss, test_accuracy = self.test()
            self.state["test_loss"].append(test_loss_sum.cpu().detach().numpy())
            self.state["test_accuracy"].append(test_accuracy.cpu().detach().numpy())

            self.scheduler.step()
            print(self.optimizer)

            print("epoch_time: {:.4f}s, epoch: {}, train_loss:{}, train_accuracy: {}".format(epoch_time, i, train_loss, train_accuracy))
            print("epoch_time: {:.4f}s, epoch: {}, test_loss:{}, test_accuracy: {}".format(epoch_time, i, test_loss, test_accuracy))

            self.model_optim_lr['model'] = self.net.state_dict()
            self.model_optim_lr['optimizer'] = self.optimizer.state_dict()
            self.model_optim_lr['lr_scheduler'] = self.scheduler
            torch.save(self.model_optim_lr, "./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".pth")

            self.state['epoch'] = i+1
            torch.save(self.state, "./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth")

            # if i % 10 == 0:
            self.plot()
            

    def test(self):
        self.net.eval()
        test_loss_sum = 0
        test_loss_avg = 0
        test_accuracy = 0
        test_total = 0
        test_correct = 0
        
        for num, (data, label) in enumerate(self.test_loader):
            data = data.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = self.net(data)
                test_loss_sum += self.criterion(output, label)

            _, predicted = torch.max(output, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum()
        

        test_accuracy = 100 * test_correct / test_total
        test_loss_avg = test_loss_sum/test_total

        print(f"test accuracy: {test_accuracy}")


        return test_loss_sum, test_loss_avg, test_accuracy

    def plot(self):
        plot_loss_accuracy("./saved_models/DenseNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth", "./plots/DenseNet/losses/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".jpg", self.trainset, self.testset)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--img_path', type=str, default='./datasets/')
    # parser.add_argument('--img_path', type=str, default='./datasets/FlowerImage')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='densenet_L100_k12')
    parser.add_argument('--scale_mode', type=str, default='single_scale')
    parser.add_argument('--aug_mode', type=str, default="")
    parser.add_argument('--weight_decay', type=float, default=0.0001)


    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()

    trainer = trainer_CIFAR10(config)
    trainer.train()
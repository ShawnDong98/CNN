import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import argparse
from tqdm import tqdm
import time


from networks.AlexNet import AlexNet, AlexNet_pytorch, AlexNet_Split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trainer():
    def __init__(self, config):
        super(trainer, self).__init__()
        self.config = config
        self.train_trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.Resize((224, 224)),
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
            # self.net = AlexNet_pytorch(num_classes=5).to(device)
            self.net = AlexNet_pytorch(num_classes=5).to(device)
            # self.net.load_state_dict(torch.load("./saved_models/AlexNet/bs20_lr004_AlexNet_WithLRN.pth"))
            # self.state = torch.load("./saved_models/AlexNet/bs20_lr004_AlexNet_WithLRN_State.pth")
            self.state = {
                "epoch": 0,
                "total_loss": [],
                "test_accuracy": []
            }
            
            print("Model loaded...")

        except:
            print("No pretrained model...")
            # self.net = AlexNet_pytorch(num_classes=5).to(device)
            self.net = AlexNet(num_classes=5).to(device)
            self.state = {
                "epoch": 0,
                "total_loss": [],
                "test_accuracy": []
            }
        

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=0.00112, momentum=0.9)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=0.01, betas=(0.9, 0.99))
        # for i in self.net.parameters():
        #     print(f"parameters: {i}")

    def train(self):
        pbar = tqdm(range(self.state["epoch"], self.config.epochs))
        for i in pbar:
            self.net.train()
            loss_sum = 0.0
            pbar.set_description(f"Now the epoch is {i}")
            start_time = time.time()
            for num, (data, label) in enumerate(self.train_loader):
                # print(data)
                # print(label)
        
                data = data.to(device)
                label = label.to(device)

                output = self.net(data)
                # print(f"output_shape: {output.shape}")
                # print(f"label: {label.shape}")

                loss = self.criterion(output, label)
                # print(loss)
                loss_sum += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # for name, parms in self.net.named_parameters():
	            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

                # print("time: {:.4f}s,  loss: {:.4f}".format(batch_time, loss))

            epoch_time = time.time() - start_time
            
            self.state["total_loss"].append(loss_sum)

            accuracy = self.test()
            self.state["test_accuracy"].append(accuracy)

            print("epoch_time: {:.4f}s, epoch: {}, loss:{}, test accuracy: {}".format(epoch_time, i, loss_sum, accuracy))

            # torch.save(self.net.state_dict(), './saved_models/AlexNet/bs20_lr004_AlexNet_WithLRN.pth')
            # self.state['epoch'] = i
            # torch.save(self.state, './saved_models/AlexNet/bs20_lr004_AlexNet_WithLRN_State.pth')

    def test(self):
        self.net.eval()
        total = 0
        correct = 0
        valid_acc_sum = 0
        n = 0
        for num, (data, label) in enumerate(self.test_loader):
            data = data.to(device)
            # print(data.shape)
            label = label.to(device)
            with torch.no_grad():
                output = self.net(data)

            # print(output)

            # print(output.shape)
            _, predicted = torch.max(output, 1)
            # print(predicted)
            total += label.size(0)
            correct += (predicted == label).sum()
        

        accuracy = 100 * correct / total

        # print(f"total: {total}")
        # print(f"correct: {correct}")
        # print(f"accuracy: {accuracy}%")

        return accuracy


class trainer_split():
    def __init__(self, config):
        super(trainer_split, self).__init__()
        self.config = config
        self.train_trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.Resize((224, 224)),
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
            # self.net = AlexNet_pytorch(num_classes=5).to(device)
            self.net = AlexNet_Split(num_classes=5).to(device)
            self.net.load_state_dict(torch.load("./saved_models/AlexNet/bs20_lr004_AlexNet_Split_WithLRN_epoch500.pth"))
            self.state = torch.load("./saved_models/AlexNet/bs20_lr004_AlexNet_Split_WithLRN_State_epoch500.pth")
            
            print("Model loaded...")

        except:
            print("No pretrained model...")
            # self.net = AlexNet_pytorch(num_classes=5).to(device)
            self.net = AlexNet_Split(num_classes=5).to(device)
            self.state = {
                "epoch": 0,
                "total_loss": [],
                "test_accuracy": []
            }
        

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=0.000112, momentum=0.9)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=0.01, betas=(0.9, 0.99))
        # for i in self.net.parameters():
        #     print(f"parameters: {i}")

    def train(self):
        pbar = tqdm(range(self.state["epoch"], self.config.epochs))
        for i in pbar:
            self.net.train()
            loss_sum = 0.0
            pbar.set_description(f"Now the epoch is {i}")
            start_time = time.time()
            for num, (data, label) in enumerate(self.train_loader):
                # print(data)
                # print(label)
                data = data.to(device)
                label = label.to(device)

                for flag in range(8):
                    output = self.net(data, flag)
                    # print(f"output_shape: {output.shape}")
                    # print(f"label: {label.shape}")

                    loss = self.criterion(output, label)
                    # print(loss)
                    loss_sum += loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # for name, parms in self.net.named_parameters():
	            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

                # print("time: {:.4f}s,  loss: {:.4f}".format(batch_time, loss))

            epoch_time = time.time() - start_time
            
            self.state["total_loss"].append(loss_sum)
            accuracy = self.test()
            self.state["test_accuracy"].append(accuracy)

            print("epoch_time: {:.4f}s, epoch: {}, loss:{}, test accuracy: {}".format(epoch_time, i, loss_sum, accuracy))

            torch.save(self.net.state_dict(), './saved_models/AlexNet/bs20_lr004_AlexNet_Split_WithLRN_epoch500.pth')
            self.state['epoch'] = i
            torch.save(self.state, './saved_models/AlexNet/bs20_lr004_AlexNet_Split_WithLRN_State_epoch500.pth')

    def test(self):
        self.net.eval()
        total = 0
        correct = 0
        
        for flag in range(8):
            best_accuracy = 0
            for num, (data, label) in enumerate(self.test_loader):
                data = data.to(device)
                label = label.to(device)
                # print(data.shape)
                with torch.no_grad():
                    output = self.net(data, flag)

                # print(output)

                # print(output.shape)
                _, predicted = torch.max(output, 1)
                # print(predicted)
                total += label.size(0)
                correct += (predicted == label).sum()
            

            accuracy = 100 * correct / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy

        # print(f"total: {total}")
        # print(f"correct: {correct}")
        # print(f"accuracy: {accuracy}%")

        return best_accuracy




def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--img_path', type=str, default='./datasets/FlowerImage/')

    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()
    trainer = trainer_split(config)
    # trainer = trainer(config)
    trainer.train()

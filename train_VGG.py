import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import argparse
import os
from tqdm import tqdm
import time


from networks.VGGNet import *
from utils.plot import plot_loss_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trainer():
    def __init__(self, config):
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
            self.net = get_vgg(self.config.model_name, num_classes=5).to(device)
            self.net.load_state_dict(torch.load("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".pth"))
            self.state = torch.load("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth")
            
            print("Model loaded...")

        except:
            print("No pretrained model...")
            self.net = get_vgg(self.config.model_name, num_classes=5).to(device)
            self.state = {
                "epoch": 0,
                "train_loss": [],
                "test_loss": [], 
                "train_accuracy": [], 
                "test_accuracy": []
            }

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.learning_rate, momentum=0.9)

       

    def train(self):
        pbar = tqdm(range(self.state["epoch"], self.config.epochs))
        for i in pbar:
            self.net.train()
            train_loss_sum = 0.0
            train_total = 0
            train_correct = 0
            pbar.set_description(f"Now the epoch is {i}")
            start_time = time.time()
            for num, (data, label) in enumerate(self.train_loader):
                data = data.to(device)
                label = label.to(device)

                output = self.net(data)

                loss = self.criterion(output, label)
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
            self.state["train_loss"].append(train_loss_sum.cpu().detach().numpy())
            self.state["train_accuracy"].append(train_accuracy.cpu().detach().numpy())
            


            test_loss_sum, test_accuracy = self.test()
            self.state["test_loss"].append(test_loss_sum.cpu().detach().numpy())
            self.state["test_accuracy"].append(test_accuracy.cpu().detach().numpy())

            print("epoch_time: {:.4f}s, epoch: {}, train_loss:{}, train_accuracy: {}".format(epoch_time, i, train_loss_sum, train_accuracy))
            print("epoch_time: {:.4f}s, epoch: {}, test_loss:{}, test_accuracy: {}".format(epoch_time, i, test_loss_sum, test_accuracy))

            torch.save(self.net.state_dict(), "./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".pth")
            self.state['epoch'] = i
            torch.save(self.state, "./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth")

            # if i % 10 == 0:
            self.plot()
            

    def test(self):
        self.net.eval()
        test_loss_sum = 0
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


        return test_loss_sum, test_accuracy

    def plot(self):
        plot_loss_accuracy("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth", "./plots/VGGNet/losses/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".jpg")




def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--img_path', type=str, default='./datasets/FlowerImage/')
    parser.add_argument('--learning_rate', type=float, default=0.00412)
    parser.add_argument('--model_name', type=str, default='vgg16_FCN')


    return parser.parse_args()



if __name__ == "__main__":
    config = get_config()
    trainer = trainer(config)
    trainer.train()  
    # trainer.plot()
    
    # conv2d = nn.Conv2d(512, 512, kernel_size=1, padding=0)  




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
from utils.dataloader import ImageLoader


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
        plot_loss_accuracy("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_State" + ".pth", "./plots/VGGNet/losses/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + ".jpg", self.trainset, self.testset)



class trainer_multi_scale():
    def __init__(self, config):
        self.config = config

        self.train_loader, self.trainset = ImageLoader(os.path.join(config.img_path, "train"), config.src_size, config.dst_size, config.batch_size, scale_mode=self.config.scale_mode, train=True, trans=True)

        self.test_loader, self.testset = ImageLoader(os.path.join(config.img_path, "test"), config.src_size, config.dst_size, 1, train=False, trans=True)

        self.net_init()

    def net_init(self):
        try:
            self.net = get_vgg(self.config.model_name, num_classes=5).to(device)

            self.net.load_state_dict(torch.load("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + str(self.config.src_size) + "_" + str(self.config.dst_size) + "_" + self.config.scale_mode + ".pth"))

            self.state = torch.load("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + str(self.config.src_size) + "_" + str(self.config.dst_size) + "_" + self.config.scale_mode + "_State" + ".pth")


            # self.net.load_state_dict(torch.load("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + "384" + "_" + "224" + ".pth"))


            # self.state = torch.load("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + "384" + "_" + "224" + "_State" + ".pth")
            
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

                loss = F.nll_loss(output, label)
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

            torch.save(self.net.state_dict(), "./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + str(self.config.src_size) + "_" + str(self.config.dst_size) + "_" + self.config.scale_mode + ".pth")

            self.state['epoch'] = i
            torch.save(self.state, "./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + str(self.config.src_size) + "_" + str(self.config.dst_size) + "_" + self.config.scale_mode + "_State" + ".pth")

            # if i % 10 == 0:
            self.plot()
            

    def test(self):
        self.net.eval()
        test_loss_sum = 0
        test_accuracy = 0
        test_total = 0
        test_correct = 0
        
        start_time = time.time()
        for num, (data, label) in enumerate(self.test_loader):
            data = data.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = self.net(data)
                test_loss_sum += F.nll_loss(output, label)

            # print(output)
            _, predicted = torch.max(output, 1)
            # print("predict: ", predicted)
            # print("label: ", label)
            test_total += label.size(0)
            test_correct += (predicted == label).sum()
        

        test_accuracy = 100 * test_correct / test_total
        consumed_time = time.time() - start_time
        print(f"time consumed: {consumed_time}")

        return test_loss_sum, test_accuracy

    def test_multi_scale(self):
        self.net.eval()
        multi_scale_loader, multi_scale_dataset = ImageLoader(os.path.join(config.img_path, "test"), 384, 224, 1, scale_mode="multi_scale", train=False)

        test_accuracy = 0
        test_total = 0
        test_correct = 0

        
        start_time = time.time()
        for datas, label in tqdm(multi_scale_loader):
            label = label.to(device)
            softmax_output_sum = 0
            for data in datas:
                data = data.to(device)
                with torch.no_grad():
                     output = self.net(data)
                     softmax_output_sum += output
                
            softmax_output_avg = softmax_output_sum / 3
            # print(softmax_output_avg)
            # print(output)
            _, predicted = torch.max(softmax_output_avg, 1)
            # print("predict: ", predicted)
            # print("label: ", label)
            test_total += label.size(0)
            test_correct += (predicted == label).sum()

        test_accuracy = 100 * test_correct / test_total
        consumed_time = time.time() - start_time
        print(f"time consumed: {consumed_time}")

        return test_accuracy

    def test_multi_crop(self):
        self.net.eval()
        multi_crop_loader, multi_crop_dataset = ImageLoader(os.path.join(config.img_path, "test"), 384, 224, 1, scale_mode="multi_crop", train=False)

        test_accuracy = 0
        test_total = 0
        test_correct = 0

        start_time = time.time()
        for datas, label in tqdm(multi_crop_loader):
            label = label
            softmax_output_sum = 0
            for data in datas:
                data = data.to(device)
                with torch.no_grad():
                     output = self.net(data)
                     softmax_output_sum += output
                
            softmax_output_avg = softmax_output_sum / 150
            # print(softmax_output_avg)
            _, predicted = torch.max(softmax_output_avg.cpu(), 1)
            test_total += 1
            test_correct += (predicted == label)

        


        test_accuracy = 100 * test_correct / test_total
        
        consumed_time = time.time() - start_time
        print(f"time consumed: {consumed_time}")

        return test_accuracy
                

    def plot(self):
        if self.config.scale_mode == 'scale_jitter':
            plot_loss_accuracy("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + str(self.config.src_size) + "_" + str(self.config.dst_size) + "_" + self.config.scale_mode + "_State" + ".pth", "./plots/VGGNet/losses/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + str(self.config.src_size) + "_" + str(self.config.dst_size) + "_" + self.config.scale_mode + ".jpg", self.trainset, self.testset)
        else:
            plot_loss_accuracy("./saved_models/VGGNet/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + str(self.config.src_size) + "_" + str(self.config.dst_size) + "_" + self.config.scale_mode + "_State" + ".pth", "./plots/VGGNet/losses/bs" + str(self.config.batch_size) + "_lr" + str(self.config.learning_rate).split('.')[1] + "_" +  str(self.config.model_name) + "_" + str(self.config.src_size) + "_" + str(self.config.dst_size) + "_" + self.config.scale_mode + ".jpg", self.trainset, self.testset)


    

def test_ConvNet_Fusion(config):
        multi_crop_loader, multi_crop_dataset = ImageLoader(os.path.join(config.img_path, "test"), 384, 224, 1, scale_mode="multi_crop", train=False)

        net = get_vgg("vgg16_multi_scale", num_classes=5).to(device)
        net.load_state_dict(torch.load("./saved_models/VGGNet/bs128_lr00412_vgg16_multi_scale_384_224_scale_jitter.pth"))
        net_FCN = get_vgg("vgg16_FCN_multi_scale", num_classes=5).to(device)
        net_FCN.load_state_dict(torch.load("./saved_models/VGGNet/bs128_lr00112_vgg16_FCN_multi_scale_384_224_scale_jitter.pth"))

        test_accuracy = 0
        test_total = 0
        test_correct = 0

        start_time = time.time()
        for datas, label in tqdm(multi_crop_loader):
            label = label
            softmax_output_sum = 0
            for data in datas:
                data = data.to(device)
                with torch.no_grad():
                     output1 = net(data)
                     softmax_output_sum += output1 / 150
                     output2 = net_FCN(data)
                     softmax_output_sum += output2 / 150
                
            softmax_output_avg = softmax_output_sum / 2
            # print(softmax_output_avg)
            _, predicted = torch.max(softmax_output_avg.cpu(), 1)
            test_total += 1
            test_correct += (predicted == label)

        


        test_accuracy = 100 * test_correct / test_total
        
        consumed_time = time.time() - start_time
        print(f"time consumed: {consumed_time}")

        return test_accuracy



def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--src_size', type=int, default=384)
    parser.add_argument('--dst_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--img_path', type=str, default='./datasets/FlowerImage/')
    parser.add_argument('--learning_rate', type=float, default=0.00112)
    parser.add_argument('--model_name', type=str, default='vgg16_FCN_multi_scale')
    parser.add_argument('--scale_mode', type=str, default='scale_jitter')


    return parser.parse_args()



if __name__ == "__main__":
    config = get_config()
    # trainer = trainer(config)
    # trainer.train()  
    # trainer.plot()
    # acc = 0
    # trainer = trainer_multi_scale(config)
    # # for i in range(10):
    # #     acc += trainer.test()[1]
    # #     # acc += trainer.test_multi_scale()
        

    # # acc = acc / 10

    # acc = trainer.test_multi_crop()
    # print(acc)

    print(test_ConvNet_Fusion(config))





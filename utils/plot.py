import torch

from torchvision import datasets


import matplotlib.pyplot as plt

import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5)
    # 没用到
    parser.add_argument('--img_size', type=int, default=256)
    # 没用到
    parser.add_argument('--total_iter', type=int, default=500000)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--img_path', type=str, default='../datasets')



    return parser.parse_args()


config = get_config()


def SameBS(bs, model):

    trainset = datasets.CIFAR10(
            root = config.img_path,
            train=True,
            download=True,
    )


    state1 = torch.load("../saved_models/LeNet/bs" +str(bs) + "_lr0004_" + model + "_CIFAR10_State.pth")
    state2 = torch.load("../saved_models/LeNet/bs" +str(bs) + "_lr004_" + model + "_CIFAR10_State.pth")
    state3 = torch.load("../saved_models/LeNet/bs" +str(bs) + "_lr04_" + model + "_CIFAR10_State.pth")

    loss1 = [x.cpu().detach().numpy()/len(trainset) for x in state1['total_loss']]
    loss2 = [x.cpu().detach().numpy()/len(trainset) for x in state2['total_loss']]
    loss3 = [x.cpu().detach().numpy()/len(trainset) for x in state3['total_loss']]

    accuracy1 = [x.cpu().detach().numpy()/100 for x in state1['test_accuracy']]
    accuracy2 = [x.cpu().detach().numpy()/100 for x in state2['test_accuracy']]
    accuracy3 = [x.cpu().detach().numpy()/100 for x in state3['test_accuracy']]

    x1 = range(0, len(loss1))
    x2 = range(0, len(loss2))
    x3 = range(0, len(loss3))

    label11 = [0.55] * len(loss1)
    label12 = [0.55] * len(loss2)
    label13 = [0.55] * len(loss3)

    label21 = [0.65] * len(loss1)
    label22 = [0.65] * len(loss2)
    label23 = [0.65] * len(loss3)

    plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(131)
    ax1.set_ylim(0, 1)
    ax1.plot(x1, loss1, label="loss")
    ax1.plot(x1, accuracy1, label="accuracy")
    ax1.plot(x1, label11, '--')
    ax1.plot(x1, label21, '--')
    ax1.legend()
    ax1.set_title(f"bs={bs} lr=0004")


    ax2 = plt.subplot(132)
    ax2.set_ylim(0, 1)
    ax2.plot(x2, loss2, label="loss")
    ax2.plot(x2, accuracy2, label="accuracy")
    ax2.plot(x2, label12, '--')
    ax2.plot(x2, label22, '--')
    ax2.legend()
    ax2.set_title(f"bs={bs} lr=004")


    ax3 = plt.subplot(133)
    ax3.set_ylim(0, 1)
    ax3.plot(x3, loss3, label="loss")
    ax3.plot(x3, accuracy3, label="accuracy")
    ax3.plot(x3, label13, '--')
    ax3.plot(x3, label23, '--')
    ax3.legend()
    ax3.set_title(f"bs={bs} lr=04")

    plt.suptitle(f"{model} model")
    plt.savefig("../plots/LeNet/losses/SameBS" + str(bs) + model  + ".jpg")
    plt.show()







def SameLR(lr, model):

    trainset = datasets.CIFAR10(
            root = config.img_path,
            train=True,
            download=True,
    )

    state1 = torch.load("../saved_models/LeNet/bs5_lr" + lr +"_" + model + "_CIFAR10_State.pth")
    state2 = torch.load("../saved_models/LeNet/bs20_lr" + lr +"_" + model + "_CIFAR10_State.pth")
    state3 = torch.load("../saved_models/LeNet/bs50_lr" + lr +"_" + model + "_CIFAR10_State.pth")
    state4 = torch.load("../saved_models/LeNet/bs100_lr" + lr +"_" + model + "_CIFAR10_State.pth")

    loss1 = [x.cpu().detach().numpy()/len(trainset) for x in state1['total_loss']]
    loss2 = [x.cpu().detach().numpy()/len(trainset) for x in state2['total_loss']]
    loss3 = [x.cpu().detach().numpy()/len(trainset) for x in state3['total_loss']]
    loss4 = [x.cpu().detach().numpy()/len(trainset) for x in state4['total_loss']]

    accuracy1 = [x.cpu().detach().numpy()/100 for x in state1['test_accuracy']]
    accuracy2 = [x.cpu().detach().numpy()/100 for x in state2['test_accuracy']]
    accuracy3 = [x.cpu().detach().numpy()/100 for x in state3['test_accuracy']]
    accuracy4 = [x.cpu().detach().numpy()/100 for x in state4['test_accuracy']]

    x1 = range(0, len(loss1))
    x2 = range(0, len(loss2))
    x3 = range(0, len(loss3))
    x4 = range(0, len(loss4))

    label11 = [0.55] * len(loss1)
    label12 = [0.55] * len(loss2)
    label13 = [0.55] * len(loss3)
    label14 = [0.55] * len(loss4)

    label21 = [0.65] * len(loss1)
    label22 = [0.65] * len(loss2)
    label23 = [0.65] * len(loss3)
    label24 = [0.65] * len(loss4)

    plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(141)
    ax1.set_ylim(0, 1)
    ax1.plot(x1, loss1, label="loss")
    ax1.plot(x1, accuracy1, label="accuracy")
    ax1.plot(x1, label11, '--')
    ax1.plot(x1, label21, '--')
    ax1.legend()
    ax1.set_title(f"bs=5 lr={lr}")


    ax2 = plt.subplot(142)
    ax2.set_ylim(0, 1)
    ax2.plot(x2, loss2, label="loss")
    ax2.plot(x2, accuracy2, label="accuracy")
    ax2.plot(x2, label12, '--')
    ax2.plot(x2, label22, '--')
    ax2.legend()
    ax2.set_title(f"bs=20 lr={lr}")


    ax3 = plt.subplot(143)
    ax3.set_ylim(0, 1)
    ax3.plot(x3, loss3, label="loss")
    ax3.plot(x3, accuracy3, label="accuracy")
    ax3.plot(x3, label13, '--')
    ax3.plot(x3, label23, '--')
    ax3.legend()
    ax3.set_title(f"bs=50 lr={lr}")


    ax4 = plt.subplot(144)
    ax4.set_ylim(0, 1)
    ax4.plot(x4, loss4, label="loss")
    ax4.plot(x4, accuracy4, label="accuracy")
    ax4.plot(x4, label14, '--')
    ax4.plot(x4, label24, '--')
    ax4.legend()
    ax4.set_title(f"bs=100 lr={lr}")

    plt.suptitle(f"{model} model")
    plt.savefig("../plots/LeNet/losses/SameLR" + lr + model  + ".jpg")
    plt.show()



if __name__ == "__main__":
    lr = "0004"
    bs = 100
    model = "LeNet"
    SameLR(lr, model)
    # SameBS(bs, model)
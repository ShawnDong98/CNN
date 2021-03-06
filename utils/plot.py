import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)

import torch

from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt

import argparse
import os


def SameBS(bs, model):

    trainset = datasets.CIFAR10(
            root = config.img_path,
            train=True,
            download=True,
    )


    state1 = torch.load("../saved_models/LeNet/bs" + str(bs) + "_lr0004_" + model + "_CIFAR10_State.pth")
    state2 = torch.load("../saved_models/LeNet/bs" + str(bs) + "_lr004_" + model + "_CIFAR10_State.pth")
    state3 = torch.load("../saved_models/LeNet/bs" + str(bs) + "_lr04_" + model + "_CIFAR10_State.pth")

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



def AlexNet_plot_loss_accuracy():

    trainset = datasets.ImageFolder(
        root = os.path.join(config.img_path, "train"),
    )


    # bs20_lr004_AlexNet_pytorch
    state_pytorch = torch.load("../saved_models/AlexNet/bs20_lr004_AlexNet_pytorch_State.pth")
    loss_pytorch = [x.cpu().detach().numpy() for x in state_pytorch['total_loss']]
    accuracy_pytorch = [x.cpu().detach().numpy()/100 for x in state_pytorch['test_accuracy']]
    x1 = range(0, len(loss_pytorch))
    label11 = [0.80] * len(loss_pytorch)
    label21 = [0.88] * len(loss_pytorch)


    # bs20_lr004_AlexNet_pytorch_WithRandomCrop
    state_pytorch_WithRandomCrop = torch.load("../saved_models/AlexNet/bs20_lr004_AlexNet_pytorch_WithRandomCrop_State.pth")
    loss_pytorch_WithRandomCrop = [x.cpu().detach().numpy() for x in state_pytorch_WithRandomCrop['total_loss']]
    accuracy_pytorch_WithRandomCrop = [x.cpu().detach().numpy()/100 for x in state_pytorch_WithRandomCrop['test_accuracy']]
    x2 = range(0, len(loss_pytorch_WithRandomCrop))
    label12 = [0.80] * len(loss_pytorch_WithRandomCrop)
    label22 = [0.88] * len(loss_pytorch_WithRandomCrop)


    # bs20_lr004_AlexNet_pytorch_WithNormalize
    state_pytorch_WithNormalize = torch.load("../saved_models/AlexNet/bs20_lr004_AlexNet_pytorch_WithNormalize_State.pth")
    loss_pytorch_WithNormalize = [x.cpu().detach().numpy() for x in state_pytorch_WithNormalize['total_loss']]
    accuracy_pytorch_WithNormalize = [x.cpu().detach().numpy()/100 for x in state_pytorch_WithNormalize['test_accuracy']]
    x3 = range(0, len(loss_pytorch_WithNormalize))
    label13 = [0.80] * len(loss_pytorch_WithNormalize)
    label23 = [0.88] * len(loss_pytorch_WithNormalize)
    

    # bs20_lr004_AlexNet_WithoutLRN
    state_MyNet_WithoutLRN = torch.load("../saved_models/AlexNet/bs20_lr004_AlexNet_WithoutLRN_State.pth")
    loss_MyNet_WithoutLRN = [x.cpu().detach().numpy() for x in state_MyNet_WithoutLRN['total_loss']]
    accuracy_MyNet_WithoutLRN = [x.cpu().detach().numpy()/100 for x in state_MyNet_WithoutLRN['test_accuracy']]
    x4 = range(0, len(loss_MyNet_WithoutLRN))
    label14 = [0.80] * len(loss_MyNet_WithoutLRN)
    label24 = [0.88] * len(loss_MyNet_WithoutLRN)





    # bs20_lr004_AlexNet_WithLRN
    state_MyNet_WithLRN = torch.load("../saved_models/AlexNet/bs20_lr004_AlexNet_WithLRN_State.pth")
    loss_MyNet_WithLRN = [x.cpu().detach().numpy() for x in state_MyNet_WithoutLRN['total_loss']]
    accuracy_MyNet_WithLRN = [x.cpu().detach().numpy()/100 for x in state_MyNet_WithLRN['test_accuracy']]
    x5 = range(0, len(loss_MyNet_WithoutLRN))
    label15 = [0.80] * len(loss_MyNet_WithoutLRN)
    label25 = [0.88] * len(loss_MyNet_WithoutLRN)


    # bs20_lr004_AlexNet_Split_WithLRN
    state_SplitNet_WithLRN = torch.load("../saved_models/AlexNet/bs20_lr004_AlexNet_Split_WithLRN_State.pth")
    loss_SplitNet_WithLRN = [x.cpu().detach().numpy() for x in state_SplitNet_WithLRN['total_loss']]
    accuracy_SplitNet_WithLRN = [x.cpu().detach().numpy()/100 for x in state_SplitNet_WithLRN['test_accuracy']]
    x6 = range(0, len(loss_SplitNet_WithLRN))
    label16 = [0.80] * len(loss_SplitNet_WithLRN)
    label26 = [0.88] * len(loss_SplitNet_WithLRN)



    plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(231)
    ax1.set_ylim(0, 1)
    ax1.plot(x1, accuracy_pytorch, color='b', label="accuracy")
    ax1.plot(x1, label11, '--')
    ax1.plot(x1, label21, '--')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('test accuracy')
    ax1.set_title(f"AlexNet Pytorch Official")
    ax11 = ax1.twinx()
    ax11.plot(x1, loss_pytorch, color='r',label="loss")
    ax11.legend(loc='lower left')


    ax2 = plt.subplot(232)
    ax2.set_ylim(0, 1)
    ax2.plot(x2, accuracy_pytorch_WithRandomCrop, color='b', label="accuracy")
    ax2.plot(x2, label12, '--')
    ax2.plot(x2, label22, '--')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('epoch')
    ax2.set_title(f"AlexNet Pytorch Official WithRandomCrop")
    ax22 = ax2.twinx()
    ax22.plot(x2, loss_pytorch_WithRandomCrop, color='r',label="loss")
    ax22.legend(loc='lower left')

    ax3 = plt.subplot(233)
    ax3.set_ylim(0, 1)
    ax3.plot(x3, accuracy_pytorch_WithNormalize, color='b', label="accuracy")
    ax3.plot(x3, label13, '--')
    ax3.plot(x3, label23, '--')
    ax3.legend(loc='upper right')
    ax3.set_xlabel('epoch')
    ax3.set_title(f"AlexNet Pytorch Official WithNormlizase")
    ax33 = ax3.twinx()
    ax33.plot(x3, loss_pytorch_WithNormalize, color='r',label="loss")
    ax33.set_ylabel('loss')
    ax33.legend(loc='lower left')

    ax4 = plt.subplot(234)
    ax4.set_ylim(0, 1)
    ax4.plot(x4, accuracy_MyNet_WithoutLRN, color='b', label="accuracy")
    ax4.plot(x4, label14, '--')
    ax4.plot(x4, label24, '--')
    ax4.legend(loc='upper right')
    ax4.set_xlabel('epoch')
    ax4.set_ylabel('test accuracy')
    ax4.set_title(f"AlexNet My Implementation")
    ax44 = ax4.twinx()
    ax44.plot(x4, loss_MyNet_WithoutLRN, color='r',label="loss")
    ax44.legend(loc='lower left')

    ax5 = plt.subplot(235)
    ax5.set_ylim(0, 1)
    ax5.plot(x5, accuracy_MyNet_WithLRN, color='b', label="accuracy")
    ax5.plot(x5, label15, '--')
    ax5.plot(x5, label25, '--')
    ax5.legend(loc='upper right')
    ax5.set_xlabel('epoch')
    ax5.set_title(f"AlexNet My Implementation WithLRN")
    ax55 = ax5.twinx()
    ax55.plot(x5, loss_MyNet_WithLRN, color='r',label="loss")
    ax55.legend(loc='lower left')

    ax6 = plt.subplot(236)
    ax6.set_ylim(0, 1)
    ax6.plot(x6, accuracy_SplitNet_WithLRN, color='b', label="accuracy")
    ax6.plot(x6, label16, '--')
    ax6.plot(x6, label26, '--')
    ax6.legend(loc='upper right')
    ax6.set_xlabel('epoch')
    ax6.set_title(f"AlexNet SplitNet WithLRN")
    ax66 = ax6.twinx()
    ax66.plot(x6, loss_SplitNet_WithLRN, color='r',label="loss")
    ax66.set_ylabel('loss')
    ax66.legend(loc='lower left')

    plt.savefig("../plots/AlexNet/losses/loss.jpg")
    plt.show()



def AlexSplit_loss_accuracy():
    trainset = datasets.ImageFolder(
        root = os.path.join(config.img_path, "train"),
    )

    state = torch.load("../saved_models/AlexNet/bs20_lr004_AlexNet_Split_WithLRN_State_epoch500.pth")

    loss = [x.cpu().detach().numpy() for x in state['total_loss']]
    accuracy = [x.cpu().detach().numpy()/100 for x in state['test_accuracy']]

    x = range(0, len(loss))
    label1 = [0.80] * len(loss)
    label2 = [0.88] * len(loss)


    plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(111)
    ax1.set_ylim(0, 1)
    ax1.plot(x, accuracy, color='b', label="accuracy")
    ax1.plot(x, label1, '--')
    ax1.plot(x, label2, '--')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('test accuracy')
    ax1.set_title(f"AlexNet Split Net")
    ax11 = ax1.twinx()
    ax11.plot(x, loss, color='r',label="loss")
    ax11.set_ylabel('loss')
    ax11.legend(loc='lower left')

    plt.savefig("../plots/AlexNet/losses/loss_split.jpg")
    plt.show()


def PlainNet_ResNet_accuracy():
    state_20 = torch.load("../saved_models/ResNet/bs128_lr00412_plain_net_20_State.pth")
    state_32 = torch.load("../saved_models/ResNet/bs128_lr00412_plain_net_32_State.pth")
    state_44 = torch.load("../saved_models/ResNet/bs128_lr00412_plain_net_44_State.pth")
    state_56 = torch.load("../saved_models/ResNet/bs128_lr00412_plain_net_56_State.pth")
    state_110 = torch.load("../saved_models/ResNet/bs128_lr00412_plain_net_110_State.pth")

    state_20_res = torch.load("../saved_models/ResNet/bs128_lr00412_resnet_CIFAR_20_State.pth")
    state_32_res = torch.load("../saved_models/ResNet/bs128_lr00412_resnet_CIFAR_32_State.pth")
    state_44_res = torch.load("../saved_models/ResNet/bs128_lr00412_resnet_CIFAR_44_State.pth")
    state_56_res = torch.load("../saved_models/ResNet/bs128_lr00412_resnet_CIFAR_56_State.pth")
    state_110_res = torch.load("../saved_models/ResNet/bs128_lr00412_resnet_CIFAR_110_State.pth")

    train_accuracy_20 = [x / 100 for x in state_20['train_accuracy']]
    train_accuracy_32 = [x / 100 for x in state_32['train_accuracy']]
    train_accuracy_44 = [x / 100 for x in state_44['train_accuracy']]
    train_accuracy_56 = [x / 100 for x in state_56['train_accuracy']]
    train_accuracy_110 = [x / 100 for x in state_110['train_accuracy']]

    test_accuracy_20 = [x / 100 for x in state_20['test_accuracy']]
    test_accuracy_32 = [x / 100 for x in state_32['test_accuracy']]
    test_accuracy_44 = [x / 100 for x in state_44['test_accuracy']]
    test_accuracy_56 = [x / 100 for x in state_56['test_accuracy']]
    test_accuracy_110 = [x / 100 for x in state_110['test_accuracy']]

    res_train_accuracy_20 = [x / 100 for x in state_20_res['train_accuracy']]
    res_train_accuracy_32 = [x / 100 for x in state_32_res['train_accuracy']]
    res_train_accuracy_44 = [x / 100 for x in state_44_res['train_accuracy']]
    res_train_accuracy_56 = [x / 100 for x in state_56_res['train_accuracy']]
    res_train_accuracy_110 = [x / 100 for x in state_110_res['train_accuracy']]

    x = range(0, len(train_accuracy_20))

    plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(121)
    ax1.set_ylim(0, 1)
    ax1.plot(x, train_accuracy_20, label="layers20")
    ax1.plot(x, train_accuracy_32, label="layers32")
    ax1.plot(x, train_accuracy_44, label="layers44")
    ax1.plot(x, train_accuracy_56, label="layers56")
    ax1.plot(x, train_accuracy_110, label="layers110")
    ax1.legend(loc='upper right')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train accuracy')
    ax1.set_title(f"Plain Net")

    ax2 = plt.subplot(122)
    ax2.set_ylim(0, 1)
    ax2.plot(x, res_train_accuracy_20, label="layers20")
    ax2.plot(x, res_train_accuracy_32, label="layers32")
    ax2.plot(x, res_train_accuracy_44, label="layers44")
    ax2.plot(x, res_train_accuracy_56, label="layers56")
    ax2.plot(x, res_train_accuracy_110, label="layers110")
    ax2.legend(loc='upper right')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('train accuracy')
    ax2.set_title(f"ResNet")

    plt.savefig("../plots/ResNet/losses/PlainNet&ResNet.jpg")
    plt.show()



def plot_loss_accuracy(filename, savefile, trainset, testset):

    state = torch.load(filename)

    train_loss = [x / len(trainset) for x in state['train_loss']]
    train_loss = np.clip(train_loss, 0, 0.1)
    test_loss =  [x / len(testset) for x in state['test_loss']]
    test_loss = np.clip(test_loss, 0, 0.1)


    train_accuracy = [x / 100 for x in state['train_accuracy']]
    test_accuracy = [x / 100 for x in state['test_accuracy']]

    # print(train_accuracy)

    x = range(0, len(train_loss))
    label = [0.80] * len(train_loss)
    label1 = [max(test_accuracy)] * len(test_accuracy)

    plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(121)
    # ax1.set_ylim(0, 1)
    ax1.plot(x, train_loss, label="train_loss")
    ax1.plot(x, test_loss,  label="test_loss")
    ax1.legend()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')


    ax2 = plt.subplot(122)
    ax2.set_ylim(0, 1)
    ax2.plot(x, train_accuracy,  label="train_accuracy")
    ax2.plot(x, test_accuracy, label="test_accuracy")
    ax2.plot(x, label, '--', label="0.8")
    ax2.plot(x, label1, '--', label="best_acc: " + str(max(test_accuracy)))
    ax2.legend(loc='lower right')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')

    plt.suptitle(filename.split("/")[-1].split('.')[-2])
    plt.savefig(savefile)
    plt.close()
    # plt.show()




def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--img_path', type=str, default='../datasets/')
    # parser.add_argument('--img_path', type=str, default='./datasets/FlowerImage')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='resnet_CIFAR_110')
    parser.add_argument('--scale_mode', type=str, default='single_scale')
    parser.add_argument('--aug_mode', type=str, default="")
    parser.add_argument('--weight_decay', type=float, default=0.0001)


    return parser.parse_args()








if __name__ == "__main__":
    # lr = "0004"
    # bs = 100
    # model = "LeNet"
    # SameLR(lr, model)
    # SameBS(bs, model)
    # AlexSplit_loss_accuracy()
    # plot_loss_accuracy("../saved_models/VGGNet/bs256_lr00412_vgg11_State.pth")
    # Plain_Net_loss_accuracy()

    config = get_config()
   
    filename = "../saved_models/ResNet/bs" + str(config.batch_size) + "_lr" + str(config.learning_rate).split('.')[1] + "_" +  str(config.model_name) + "_State" + ".pth"

    saved_path = "../plots/ResNet/losses/bs" + str(config.batch_size) + "_lr" + str(config.learning_rate).split('.')[1] + "_" +  str(config.model_name) + ".jpg"


    # CIFAR10
    trainset = datasets.CIFAR10(
        root = config.img_path,
        train=True,
        download=True,
    )
    
    # CIFAR10
    testset = datasets.CIFAR10(
        root = config.img_path,
        train=False,
        download=True,
    )



    plot_loss_accuracy(filename, saved_path, trainset, testset)
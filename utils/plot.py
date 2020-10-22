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


# trainset = datasets.CIFAR10(
#             root = config.img_path,
#             train=True,
#             download=True,
#         )


# state_bs5_lr0004_Linear = torch.load("../saved_models/LeNet/bs5_lr0004_Linear_CIFAR10_State.pth")
# state_bs20_lr0004_Linear = torch.load("../saved_models/LeNet/bs20_lr0004_Linear_CIFAR10_State.pth")
# state_bs50_lr0004_Linear = torch.load("../saved_models/LeNet/bs50_lr0004_Linear_CIFAR10_State.pth")
# state_bs100_lr0004_Linear = torch.load("../saved_models/LeNet/bs100_lr0004_Linear_CIFAR10_State.pth")

# loss_bs5_lr0004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs5_lr0004_Linear['total_loss']]
# loss_bs20_lr0004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs20_lr0004_Linear['total_loss']]
# loss_bs50_lr0004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs50_lr0004_Linear['total_loss']]
# loss_bs100_lr0004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs100_lr0004_Linear['total_loss']]



# accuracy_bs5_lr0004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs5_lr0004_Linear['test_accuracy']]
# accuracy_bs20_lr0004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs20_lr0004_Linear['test_accuracy']]
# accuracy_bs50_lr0004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs50_lr0004_Linear['test_accuracy']]
# accuracy_bs100_lr0004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs100_lr0004_Linear['test_accuracy']]



# x_bs5_lr0004_Linear = range(0, len(loss_bs5_lr0004_Linear))
# x_bs20_lr0004_Linear = range(0, len(loss_bs20_lr0004_Linear))
# x_bs50_lr0004_Linear = range(0, len(loss_bs50_lr0004_Linear))
# x_bs100_lr0004_Linear = range(0, len(loss_bs100_lr0004_Linear))



# plt.figure(figsize=(20, 10))

# ax1 = plt.subplot(141)
# ax1.set_ylim(0, 1)
# ax1.plot(x_bs5_lr0004_Linear, loss_bs5_lr0004_Linear, label="loss")
# ax1.plot(x_bs5_lr0004_Linear, accuracy_bs5_lr0004_Linear, label="accuracy")
# ax1.legend()
# ax1.set_title("bs=5 lr=0.0004")


# ax2 = plt.subplot(142)
# ax2.set_ylim(0, 1)
# ax2.plot(x_bs20_lr0004_Linear, loss_bs20_lr0004_Linear, label="loss")
# ax2.plot(x_bs20_lr0004_Linear, accuracy_bs20_lr0004_Linear, label="accuracy")
# ax2.legend()
# ax2.set_title("bs=20 lr=0.0004")


# ax3 = plt.subplot(143)
# ax3.set_ylim(0, 1)
# ax3.plot(x_bs50_lr0004_Linear, loss_bs50_lr0004_Linear, label="loss")
# ax3.plot(x_bs50_lr0004_Linear, accuracy_bs50_lr0004_Linear, label="accuracy")
# ax3.legend()
# ax3.set_title("bs=50 lr=0.0004")


# ax4 = plt.subplot(144)
# ax4.set_ylim(0, 1)
# ax4.plot(x_bs100_lr0004_Linear, loss_bs100_lr0004_Linear, label="loss")
# ax4.plot(x_bs100_lr0004_Linear, accuracy_bs100_lr0004_Linear, label="accuracy")
# ax4.legend()
# ax4.set_title("bs=100 lr=0.0004")

# plt.suptitle("Linear model")
# plt.savefig("../plots/LeNet/losses/Linear.jpg")
# plt.show()

# state_bs5_lr0004_LeNet = torch.load("../saved_models/LeNet/bs5_lr0004_LeNet_CIFAR10_State.pth")
# state_bs20_lr0004_LeNet = torch.load("../saved_models/LeNet/bs20_lr0004_LeNet_CIFAR10_State.pth")
# state_bs50_lr0004_LeNet = torch.load("../saved_models/LeNet/bs50_lr0004_LeNet_CIFAR10_State.pth")
# state_bs100_lr0004_LeNet = torch.load("../saved_models/LeNet/bs100_lr0004_LeNet_CIFAR10_State.pth")

# loss_bs5_lr0004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs5_lr0004_LeNet['total_loss']]
# loss_bs20_lr0004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs20_lr0004_LeNet['total_loss']]
# loss_bs50_lr0004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs50_lr0004_LeNet['total_loss']]
# loss_bs100_lr0004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs100_lr0004_LeNet['total_loss']]



# accuracy_bs5_lr0004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs5_lr0004_LeNet['test_accuracy']]
# accuracy_bs20_lr0004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs20_lr0004_LeNet['test_accuracy']]
# accuracy_bs50_lr0004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs50_lr0004_LeNet['test_accuracy']]
# accuracy_bs100_lr0004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs100_lr0004_LeNet['test_accuracy']]



# x_bs5_lr0004_LeNet = range(0, len(loss_bs5_lr0004_LeNet))
# x_bs20_lr0004_LeNet = range(0, len(loss_bs20_lr0004_LeNet))
# x_bs50_lr0004_LeNet = range(0, len(loss_bs50_lr0004_LeNet))
# x_bs100_lr0004_LeNet = range(0, len(loss_bs100_lr0004_LeNet))



# plt.figure(figsize=(20, 10))

# ax1 = plt.subplot(141)
# ax1.set_ylim(0, 1)
# ax1.plot(x_bs5_lr0004_LeNet, loss_bs5_lr0004_LeNet, label="loss")
# ax1.plot(x_bs5_lr0004_LeNet, accuracy_bs5_lr0004_LeNet, label="accuracy")
# ax1.legend()
# ax1.set_title("bs=5 lr=0.0004")


# ax2 = plt.subplot(142)
# ax2.set_ylim(0, 1)
# ax2.plot(x_bs20_lr0004_LeNet, loss_bs20_lr0004_LeNet, label="loss")
# ax2.plot(x_bs20_lr0004_LeNet, accuracy_bs20_lr0004_LeNet, label="accuracy")
# ax2.legend()
# ax2.set_title("bs=20 lr=0.0004")


# ax3 = plt.subplot(143)
# ax3.set_ylim(0, 1)
# ax3.plot(x_bs50_lr0004_LeNet, loss_bs50_lr0004_LeNet, label="loss")
# ax3.plot(x_bs50_lr0004_LeNet, accuracy_bs50_lr0004_LeNet, label="accuracy")
# ax3.legend()
# ax3.set_title("bs=50 lr=0.0004")


# ax4 = plt.subplot(144)
# ax4.set_ylim(0, 1)
# ax4.plot(x_bs100_lr0004_LeNet, loss_bs100_lr0004_LeNet, label="loss")
# ax4.plot(x_bs100_lr0004_LeNet, accuracy_bs100_lr0004_LeNet, label="accuracy")
# ax4.legend()
# ax4.set_title("bs=100 lr=0.0004")

# plt.suptitle("LeNet model")
# plt.savefig("../plots/LeNet/losses/LeNet.jpg")
# plt.show()


# state_bs5_lr004_LeNet = torch.load("../saved_models/LeNet/bs5_lr004_LeNet_CIFAR10_State.pth")
# state_bs20_lr004_LeNet = torch.load("../saved_models/LeNet/bs20_lr004_LeNet_CIFAR10_State.pth")
# state_bs50_lr004_LeNet = torch.load("../saved_models/LeNet/bs50_lr004_LeNet_CIFAR10_State.pth")
# state_bs100_lr004_LeNet = torch.load("../saved_models/LeNet/bs100_lr004_LeNet_CIFAR10_State.pth")

# loss_bs5_lr004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs5_lr004_LeNet['total_loss']]
# loss_bs20_lr004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs20_lr004_LeNet['total_loss']]
# loss_bs50_lr004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs50_lr004_LeNet['total_loss']]
# loss_bs100_lr004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs100_lr004_LeNet['total_loss']]



# accuracy_bs5_lr004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs5_lr004_LeNet['test_accuracy']]
# accuracy_bs20_lr004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs20_lr004_LeNet['test_accuracy']]
# accuracy_bs50_lr004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs50_lr004_LeNet['test_accuracy']]
# accuracy_bs100_lr004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs100_lr004_LeNet['test_accuracy']]



# x_bs5_lr004_LeNet = range(0, len(loss_bs5_lr004_LeNet))
# x_bs20_lr004_LeNet = range(0, len(loss_bs20_lr004_LeNet))
# x_bs50_lr004_LeNet = range(0, len(loss_bs50_lr004_LeNet))
# x_bs100_lr004_LeNet = range(0, len(loss_bs100_lr004_LeNet))



# plt.figure(figsize=(20, 10))

# ax1 = plt.subplot(141)
# ax1.set_ylim(0, 1)
# ax1.plot(x_bs5_lr004_LeNet, loss_bs5_lr004_LeNet, label="loss")
# ax1.plot(x_bs5_lr004_LeNet, accuracy_bs5_lr004_LeNet, label="accuracy")
# ax1.legend()
# ax1.set_title("bs=5 lr=0.004")


# ax2 = plt.subplot(142)
# ax2.set_ylim(0, 1)
# ax2.plot(x_bs20_lr004_LeNet, loss_bs20_lr004_LeNet, label="loss")
# ax2.plot(x_bs20_lr004_LeNet, accuracy_bs20_lr004_LeNet, label="accuracy")
# ax2.legend()
# ax2.set_title("bs=20 lr=0.004")


# ax3 = plt.subplot(143)
# ax3.set_ylim(0, 1)
# ax3.plot(x_bs50_lr004_LeNet, loss_bs50_lr004_LeNet, label="loss")
# ax3.plot(x_bs50_lr004_LeNet, accuracy_bs50_lr004_LeNet, label="accuracy")
# ax3.legend()
# ax3.set_title("bs=50 lr=0.004")


# ax4 = plt.subplot(144)
# ax4.set_ylim(0, 1)
# ax4.plot(x_bs100_lr004_LeNet, loss_bs100_lr004_LeNet, label="loss")
# ax4.plot(x_bs100_lr004_LeNet, accuracy_bs100_lr004_LeNet, label="accuracy")
# ax4.legend()
# ax4.set_title("bs=100 lr=0.004")

# plt.suptitle("LeNet model")
# plt.savefig("../plots/LeNet/losses/LeNet.jpg")
# plt.show()




# state_bs5_lr004_Linear = torch.load("../saved_models/LeNet/bs5_lr004_Linear_CIFAR10_State.pth")
# state_bs20_lr004_Linear = torch.load("../saved_models/LeNet/bs20_lr004_Linear_CIFAR10_State.pth")
# state_bs50_lr004_Linear = torch.load("../saved_models/LeNet/bs50_lr004_Linear_CIFAR10_State.pth")
# state_bs100_lr004_Linear = torch.load("../saved_models/LeNet/bs100_lr004_Linear_CIFAR10_State.pth")

# loss_bs5_lr004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs5_lr004_Linear['total_loss']]
# loss_bs20_lr004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs20_lr004_Linear['total_loss']]
# loss_bs50_lr004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs50_lr004_Linear['total_loss']]
# loss_bs100_lr004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs100_lr004_Linear['total_loss']]



# accuracy_bs5_lr004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs5_lr004_Linear['test_accuracy']]
# accuracy_bs20_lr004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs20_lr004_Linear['test_accuracy']]
# accuracy_bs50_lr004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs50_lr004_Linear['test_accuracy']]
# accuracy_bs100_lr004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs100_lr004_Linear['test_accuracy']]



# x_bs5_lr004_Linear = range(0, len(loss_bs5_lr004_Linear))
# x_bs20_lr004_Linear = range(0, len(loss_bs20_lr004_Linear))
# x_bs50_lr004_Linear = range(0, len(loss_bs50_lr004_Linear))
# x_bs100_lr004_Linear = range(0, len(loss_bs100_lr004_Linear))



# plt.figure(figsize=(20, 10))

# ax1 = plt.subplot(141)
# ax1.set_ylim(0, 1)
# ax1.plot(x_bs5_lr004_Linear, loss_bs5_lr004_Linear, label="loss")
# ax1.plot(x_bs5_lr004_Linear, accuracy_bs5_lr004_Linear, label="accuracy")
# ax1.legend()
# ax1.set_title("bs=5 lr=0.004")


# ax2 = plt.subplot(142)
# ax2.set_ylim(0, 1)
# ax2.plot(x_bs20_lr004_Linear, loss_bs20_lr004_Linear, label="loss")
# ax2.plot(x_bs20_lr004_Linear, accuracy_bs20_lr004_Linear, label="accuracy")
# ax2.legend()
# ax2.set_title("bs=20 lr=0.004")


# ax3 = plt.subplot(143)
# ax3.set_ylim(0, 1)
# ax3.plot(x_bs50_lr004_Linear, loss_bs50_lr004_Linear, label="loss")
# ax3.plot(x_bs50_lr004_Linear, accuracy_bs50_lr004_Linear, label="accuracy")
# ax3.legend()
# ax3.set_title("bs=50 lr=0.004")


# ax4 = plt.subplot(144)
# ax4.set_ylim(0, 1)
# ax4.plot(x_bs100_lr004_Linear, loss_bs100_lr004_Linear, label="loss")
# ax4.plot(x_bs100_lr004_Linear, accuracy_bs100_lr004_Linear, label="accuracy")
# ax4.legend()
# ax4.set_title("bs=100 lr=0.004")

# plt.suptitle("Linear model")
# plt.savefig("../plots/LeNet/losses/Linear.jpg")
# plt.show()


# state_bs5_lr04_Linear = torch.load("../saved_models/LeNet/bs5_lr04_Linear_CIFAR10_State.pth")
# state_bs20_lr04_Linear = torch.load("../saved_models/LeNet/bs20_lr04_Linear_CIFAR10_State.pth")
# state_bs50_lr04_Linear = torch.load("../saved_models/LeNet/bs50_lr04_Linear_CIFAR10_State.pth")
# state_bs100_lr04_Linear = torch.load("../saved_models/LeNet/bs100_lr04_Linear_CIFAR10_State.pth")

# loss_bs5_lr04_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs5_lr04_Linear['total_loss']]
# loss_bs20_lr04_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs20_lr04_Linear['total_loss']]
# loss_bs50_lr04_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs50_lr04_Linear['total_loss']]
# loss_bs100_lr04_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs100_lr04_Linear['total_loss']]



# accuracy_bs5_lr04_Linear = [x.cpu().detach().numpy()/100 for x in state_bs5_lr04_Linear['test_accuracy']]
# accuracy_bs20_lr04_Linear = [x.cpu().detach().numpy()/100 for x in state_bs20_lr04_Linear['test_accuracy']]
# accuracy_bs50_lr04_Linear = [x.cpu().detach().numpy()/100 for x in state_bs50_lr04_Linear['test_accuracy']]
# accuracy_bs100_lr04_Linear = [x.cpu().detach().numpy()/100 for x in state_bs100_lr04_Linear['test_accuracy']]



# x_bs5_lr04_Linear = range(0, len(loss_bs5_lr04_Linear))
# x_bs20_lr04_Linear = range(0, len(loss_bs20_lr04_Linear))
# x_bs50_lr04_Linear = range(0, len(loss_bs50_lr04_Linear))
# x_bs100_lr04_Linear = range(0, len(loss_bs100_lr04_Linear))



# plt.figure(figsize=(20, 10))

# ax1 = plt.subplot(141)
# ax1.set_ylim(0, 1)
# ax1.plot(x_bs5_lr04_Linear, loss_bs5_lr04_Linear, label="loss")
# ax1.plot(x_bs5_lr04_Linear, accuracy_bs5_lr04_Linear, label="accuracy")
# ax1.legend()
# ax1.set_title("bs=5 lr=0.04")


# ax2 = plt.subplot(142)
# ax2.set_ylim(0, 1)
# ax2.plot(x_bs20_lr04_Linear, loss_bs20_lr04_Linear, label="loss")
# ax2.plot(x_bs20_lr04_Linear, accuracy_bs20_lr04_Linear, label="accuracy")
# ax2.legend()
# ax2.set_title("bs=20 lr=0.04")


# ax3 = plt.subplot(143)
# ax3.set_ylim(0, 1)
# ax3.plot(x_bs50_lr04_Linear, loss_bs50_lr04_Linear, label="loss")
# ax3.plot(x_bs50_lr04_Linear, accuracy_bs50_lr04_Linear, label="accuracy")
# ax3.legend()
# ax3.set_title("bs=50 lr=0.04")


# ax4 = plt.subplot(144)
# ax4.set_ylim(0, 1)
# ax4.plot(x_bs100_lr04_Linear, loss_bs100_lr04_Linear, label="loss")
# ax4.plot(x_bs100_lr04_Linear, accuracy_bs100_lr04_Linear, label="accuracy")
# ax4.legend()
# ax4.set_title("bs=100 lr=0.04")

# plt.suptitle("Linear model")
# plt.savefig("../plots/LeNet/losses/Linear.jpg")
# plt.show()


# state_bs5_lr04_LeNet = torch.load("../saved_models/LeNet/bs5_lr04_LeNet_CIFAR10_State.pth")
# state_bs20_lr04_LeNet = torch.load("../saved_models/LeNet/bs20_lr04_LeNet_CIFAR10_State.pth")
# state_bs50_lr04_LeNet = torch.load("../saved_models/LeNet/bs50_lr04_LeNet_CIFAR10_State.pth")
# state_bs100_lr04_LeNet = torch.load("../saved_models/LeNet/bs100_lr04_LeNet_CIFAR10_State.pth")

# loss_bs5_lr04_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs5_lr04_LeNet['total_loss']]
# loss_bs20_lr04_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs20_lr04_LeNet['total_loss']]
# loss_bs50_lr04_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs50_lr04_LeNet['total_loss']]
# loss_bs100_lr04_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs100_lr04_LeNet['total_loss']]



# accuracy_bs5_lr04_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs5_lr04_LeNet['test_accuracy']]
# accuracy_bs20_lr04_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs20_lr04_LeNet['test_accuracy']]
# accuracy_bs50_lr04_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs50_lr04_LeNet['test_accuracy']]
# accuracy_bs100_lr04_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs100_lr04_LeNet['test_accuracy']]



# x_bs5_lr04_LeNet = range(0, len(loss_bs5_lr04_LeNet))
# x_bs20_lr04_LeNet = range(0, len(loss_bs20_lr04_LeNet))
# x_bs50_lr04_LeNet = range(0, len(loss_bs50_lr04_LeNet))
# x_bs100_lr04_LeNet = range(0, len(loss_bs100_lr04_LeNet))



# plt.figure(figsize=(20, 10))

# ax1 = plt.subplot(141)
# ax1.set_ylim(0, 1)
# ax1.plot(x_bs5_lr04_LeNet, loss_bs5_lr04_LeNet, label="loss")
# ax1.plot(x_bs5_lr04_LeNet, accuracy_bs5_lr04_LeNet, label="accuracy")
# ax1.legend()
# ax1.set_title("bs=5 lr=0.04")


# ax2 = plt.subplot(142)
# ax2.set_ylim(0, 1)
# ax2.plot(x_bs20_lr04_LeNet, loss_bs20_lr04_LeNet, label="loss")
# ax2.plot(x_bs20_lr04_LeNet, accuracy_bs20_lr04_LeNet, label="accuracy")
# ax2.legend()
# ax2.set_title("bs=20 lr=0.04")


# ax3 = plt.subplot(143)
# ax3.set_ylim(0, 1)
# ax3.plot(x_bs50_lr04_LeNet, loss_bs50_lr04_LeNet, label="loss")
# ax3.plot(x_bs50_lr04_LeNet, accuracy_bs50_lr04_LeNet, label="accuracy")
# ax3.legend()
# ax3.set_title("bs=50 lr=0.04")


# ax4 = plt.subplot(144)
# ax4.set_ylim(0, 1)
# ax4.plot(x_bs100_lr04_LeNet, loss_bs100_lr04_LeNet, label="loss")
# ax4.plot(x_bs100_lr04_LeNet, accuracy_bs100_lr04_LeNet, label="accuracy")
# ax4.legend()
# ax4.set_title("bs=100 lr=0.04")

# plt.suptitle("LeNet model")
# plt.savefig("../plots/LeNet/losses/LeNet.jpg")
# plt.show()


trainset = datasets.MNIST(
            root = config.img_path,
            train=True,
            download=True,
        )

state_bs100_lr004_LeNet = torch.load("../saved_models/LeNet/bs100_lr004_LeNet_MNIST_State.pth")
state_bs100_lr004_Linear = torch.load("../saved_models/LeNet/bs100_lr004_Linear_MNIST_State.pth")

loss_bs100_lr004_LeNet = [x.cpu().detach().numpy()/len(trainset) for x in state_bs100_lr004_LeNet['total_loss']]
loss_bs100_lr004_Linear = [x.cpu().detach().numpy()/len(trainset) for x in state_bs100_lr004_Linear['total_loss']]


accuracy_bs100_lr004_LeNet = [x.cpu().detach().numpy()/100 for x in state_bs100_lr004_LeNet['test_accuracy']]
accuracy_bs100_lr004_Linear = [x.cpu().detach().numpy()/100 for x in state_bs100_lr004_Linear['test_accuracy']]


x_bs100_lr004_LeNet = range(0, len(loss_bs100_lr004_LeNet))
x_bs100_lr004_Linear = range(0, len(loss_bs100_lr004_Linear))

plt.figure(figsize=(20, 10))

ax1 = plt.subplot(121)
ax1.set_ylim(0, 1)
ax1.plot(x_bs100_lr004_LeNet, loss_bs100_lr004_LeNet, label="loss")
ax1.plot(x_bs100_lr004_LeNet, accuracy_bs100_lr004_LeNet, label="accuracy")
ax1.legend()
ax1.set_title("LeNet model")


ax2 = plt.subplot(122)
ax2.set_ylim(0, 1)
ax2.plot(x_bs100_lr004_LeNet, loss_bs100_lr004_Linear, label="loss")
ax2.plot(x_bs100_lr004_LeNet, accuracy_bs100_lr004_Linear, label="accuracy")
ax2.legend()
ax2.set_title("Linear model")

plt.savefig("../plots/LeNet/losses/MNIST.jpg")
plt.show()


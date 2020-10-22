import cv2
import numpy as np


def concat2pic():
    img1 = cv2.imread("../plots/LeNet/losses/Linear.jpg")
    img2 = cv2.imread("../plots/LeNet/losses/LeNet.jpg")

    print(img1.shape)
    print(img2.shape)

    lr04 = np.concatenate([img1, img2], axis=0)

    # cv2.imshow("merge", merge)
    cv2.imwrite("../plots/LeNet/losses/lr04.jpg", lr04)


if __name__ == "__main__":
    concat2pic()
        
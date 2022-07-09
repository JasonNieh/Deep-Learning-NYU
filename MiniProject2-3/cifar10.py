import pickle as p
import numpy as np
from PIL import Image
import os


clslist = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='latin1')
        print(datadict.keys())
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


if __name__ == "__main__":
    imgX, imgY = load_CIFAR_batch(r'.\CIFAR10\cifar-10-batches-py\test_batch')
    save_path = r'.\datasets\cifar_test'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(imgX.shape[0]):
        imgs = imgX[i - 1]
        clsindex = imgY[i - 1]
        save_sub_path = os.path.join(save_path, clslist[clsindex])
        if not os.path.exists(save_sub_path):
            os.mkdir(save_sub_path)

        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB",(i0,i1,i2))
        name = os.path.join(save_sub_path, "batch5_" + str(i) + ".png")
        img.save(name)

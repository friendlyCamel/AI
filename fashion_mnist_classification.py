import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch import nn
from IPython import display
from torch.utils import data
from d2l import torch as d2l
from source_code.timer import Timer
d2l.use_svg_display()

# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(
#     root="../ATRY_PROJECT/mnist_data", train=True, transform=trans, download=True
# )
# mnist_test = torchvision.datasets.FashionMNIST(
#     root="/ATRY_PROJECT/mnist_data", train=False, transform=trans, download=True
# )
# print(len(mnist_test), len(mnist_train))
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, title=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if title:
            ax.set_title(title[i])
    return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, title=get_fashion_mnist_labels(y))
# plt.show()

# batch_size = 256
def get_dataloader_workers():
    return 4
# train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
# timer = d2l.Timer()
# for X, y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec')

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="../ATRY_PROJECT/mnist_data", train=True, transform=trans, download=True
        )
        mnist_test = torchvision.datasets.FashionMNIST(
            root="/ATRY_PROJECT/mnist_data", train=False, transform=trans, download=True
        )
        return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers())
                )

# train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     show_images(X.reshape(32, 64, 64), 4, 8, title=get_fashion_mnist_labels(y))
#     plt.show()
#     break
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 50

def train_epoch_ch3(net, train_iter,loss, updater):
    net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter,loss,updater)
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        print(train_metrics)
        print(test_acc)
        # train_loss, train_acc = train_metrics
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

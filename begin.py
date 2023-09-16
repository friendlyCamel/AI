"""
    2023年9月16日
"""
import math
import random
import numpy as np
import torch
from d2l import torch as d2l
from source_code.timer import Timer
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 /sigma**2 * (x - mu)**2)

# # 对比不同的高斯分布
# x = np.arange(-7, 7, 0.01)
# params = [(0, 1), (0, 2), (3, 1)]
# d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
#          ylabel='p(x)', figsize=(4.5, 2.5),
#          legend=[f'mean{mu}, std{sigma}' for mu, sigma in params])
# d2l.plt.show()
# 合成数据集
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, label = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', label[0])


d2l.set_figsize()
d2l.plt.scatter(features[:, 0].detach().numpy(), label.detach().numpy(), 1)
d2l.plt.show()

def data_iter(batch_size, features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    return torch.matmul(X, w) + b
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.3
num_epochs = 50
net = linreg
loss = squared_loss
batch_size = 1000
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, label):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), label)
        print(f"epoch{epoch + 1}, loss {float(train_l.mean()):f}")

print(f"w的估计误差: {true_w - w.reshape(true_w.shape)}")
print(f"b的估计误差: {true_b - b}")
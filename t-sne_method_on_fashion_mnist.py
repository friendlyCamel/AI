import torch
from d2l import torch as d2l
import numpy as np
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)



def load(train_iter):
    data = np.zeros((1, 784))
    label = np.zeros((1, 1))
    i = 0
    for X, y in tqdm(train_iter):
        # data.append(X.numpy().reshape(-1, 784))
        print(data.shape)
        print(X.reshape(-1, 784)[0].shape)
        data = np.concatenate((data, X.numpy().reshape((-1, 784))), axis=0)
        # label.append(y.numpy())
        label = np.concatenate((label, y.numpy().reshape((-1, 1))), axis=0)
        i += 1
        if i >= 1:
            break
    print(np.asarray(data).shape)
    print(type(np.asarray(data)))
    data = (np.asarray(data).reshape(257, 784))
    label = np.asarray(label).reshape(257, 1)
    return data, label


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

if __name__ == "__main__":
    # 做了下t-sne不过感觉效果不明显啊
    data, label = load(train_iter)
    data_array = data.reshape(257, 784)
    label_array = label.reshape(257, 1)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=500, perplexity=30, n_iter=10000, learning_rate=200)
    result_2d = tsne.fit_transform(data_array)
    fig1 = plot_embedding_2D(result_2d, label_array, "per:50")
    fig1.show()

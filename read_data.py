#读取mnist数据集
import os
import numpy as np

def read_data():

    data_dir = 'D:\deeplearning\GAN\MNIST_data'

    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    #转换成numpy数据
    loaded = np.fromfile(file = fd, dtype=np.uint8)
    #根据 mnist 官网描述的数据格式，图像像素从 16 字节开始
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file = fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.uint8)

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file = fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file = fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.uint8)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    # 由于生成网络由服从某一分布的噪声生成图片，不需要测试集，
    # 所以把训练和测试两部分数据合并
    X = np.concatenate((trX, teX), axis=0)
    Y = np.concatenate((trY, teY), axis=0)

    # 打乱排序
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)

    #y_vec 表示对网络所加的约束条件，这个条件是类别标签
    #y_vec 实际就是对 y 的独热编码
    y_vec = np.zeros((len(Y), 10), dtype=np.uint8)
    for i, label in enumerate(Y):
        y_vec[i, Y[i]] = 1.0

    return X/255, y_vec

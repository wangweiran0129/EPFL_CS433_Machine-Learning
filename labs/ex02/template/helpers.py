# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def load_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metrics system."""
    path_dataset = "height_weight_genders.csv"
    # use np.genfromtxt to import data
    # delimiter 分隔符
    # skip_header 跳过特定行
    # usecols 选择特定的行
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50] # extract the first 50 values in height
        weight = weight[::50] # extract the first 50 values in weight
        print("The shape of height is ", np.shape(height)) #(200, )
        print("The shape of weight is ", np.shape(weight)) #(200, )

    # 增加异常值
    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y) # 10000
    # put 1 as the 0th column
    tx = np.c_[np.ones(num_samples), x]
    """
    Example np.c_
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.c_[a, b]
    [[1, 4]
    [2, 5]
    [3, 6]]
    """
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y) #10000
    
    # 打乱顺序
    if shuffle:
        # np.random.permutation 生成随机数列
        # shuffle_indices 就是将0 - 9999打乱后生成一个随机数列
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # y和tx是原数列，但打乱了顺序
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    # 保持原来顺序
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            # yield可以理解为return，但程序下次会从这里再次运行
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

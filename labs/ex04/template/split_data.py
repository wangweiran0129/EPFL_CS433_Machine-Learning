# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    # raise NotImplementedError
    
    # Change the order of samples
    index = np.random.permutation(x.shape[0])
    split_location = int(np.floor(ratio * x.shape[0]))
    
    index_train = index[:split_location]
    index_test = index[split_location:]
    
    x_train = x[index_train]
    x_test = x[index_test]
    
    y_train = y[index_train]
    y_test = y[index_test]
    
    return x_train, y_train, x_test, y_test

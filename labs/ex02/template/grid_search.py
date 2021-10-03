# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
from costs import *
import helpers


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    # losses is a matrix full of 0
    losses = np.zeros((len(w0), len(w1)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    
    for idx_w0, w0_val in enumerate(w0):
        for idx_w1, w1_val in enumerate(w1):
            w = np.array([w0_val, w1_val])
            losses[idx_w0, idx_w1] = compute_loss(y, tx, w)
    
    #raise NotImplementedError
    return losses

def generate_w(num_intervals):
    # np.linspace 在指定的间隔内返回均匀间隔的数字
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    # np.unravel_index
    # 求出数组某元素（或某组元素）拉成一维后的索引值在原本维度（或指定新维度）中对应的索引
    # np.argmin() -> 给出最小值的下标
    # min_row -> losses中最小值的行
    # min_col -> losses中最小值的列
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    
    # losses[min_row, min_col] 是losses中的最小值
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search
#       here when it is done.
# ***************************************************


# key: how to find the losses? -> losses: (np.shape(y))

'''
num_intervals = 2
w0 = [-100, 200]
w1 = [-150, 150]

w0, w1 = generate_w(10)
losses_list = []

(height, weight, gender) = helpers.load_data(sub_sample=True, add_outlier=False)
(y,tx) = helpers.build_model_data(height, weight)

for i in w0:
    for j in w1:
        loss = costs.compute_loss(y, tx, [i, j])
        print(loss)
        losses_list.append(loss)
        
losses = np.array(losses_list).reshape(len(w0),len(w1))
print(get_best_parameters(w0, w1, losses))
'''

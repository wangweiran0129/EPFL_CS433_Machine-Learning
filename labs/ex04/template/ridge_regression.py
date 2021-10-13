# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import *


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    #raise NotImplementedError
    
    # left_part * w_ridge = right_part
    left_part = np.dot(tx.T, tx) + lambda_ * 2 * np.shape(tx)[0] * np.eye(np.shape(tx)[1])
    right_part = np.dot(tx.T, y)
    
    w_ridge = np.linalg.solve(left_part, right_part)
    # w_ridge = np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_), np.dot(tx.T, y))
    # mse_ridge = compute_mse(y, tx, w_ridge)
    
    # return mse_ridge, w_ridge
    return w_ridge

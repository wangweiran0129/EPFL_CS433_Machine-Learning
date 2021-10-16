# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns optimal weights, MSE
    # ***************************************************
    # raise NotImplementedError
    
    optimal_w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    
    # compute_loss(y, tx, w, flag = 0)
    mse = compute_mse(y, tx, optimal_w)
    
    return mse, optimal_w

# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    #raise NotImplementedError
    
    ones = np.ones([np.shape(x)[0], 1])
    
    phi_xn = np.zeros([np.shape(x)[0], degree])
    for i in range(np.shape(x)[0]):
        for j in range(degree):
            phi_xn[i][j] = pow(x[i], j+1)
            
    phi = np.hstack((ones, phi_xn))
            
    return phi

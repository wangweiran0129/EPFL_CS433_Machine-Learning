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
    # raise NotImplementedError

    phi_xn = np.zeros([np.shape(x)[0], degree+1])
    
    for i in range(np.shape(x)[0]):
        for j in range(degree+1):
            phi_xn[i][j] = pow(x[i], j)
            
    # phi = np.hstack((ones, phi_xn))
            
    return phi_xn
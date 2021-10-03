# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np
import costs

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and loss
    # ***************************************************
    #raise NotImplementedError
    
    e = y - np.dot(tx, w)
    #print(e)
    N = np.shape(y)[0]
    
    # Mean Square Error
    gradient_mse = (-1/N) * np.dot(tx.T, e)
    return gradient_mse
    
    '''
    # Mean Absolute Error
    gradient_mae = (-1/N) * np.dot(tx.T, e)
    return gradient_mae
    '''

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        loss = costs.compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)
        #raise NotImplementedError
        
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma * grad 
        #raise NotImplementedError
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

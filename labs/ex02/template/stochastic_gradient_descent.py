# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import gradient_descent
import helpers
import costs

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    
    grad = gradient_descent.compute_gradient(y, tx, w)
    return grad
    
    #raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    
    ws = [initial_w]
    losses = []
    w = initial_w
    
    # def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
    for n_iter in range(max_iters):
        for y_batch, tx_batch in helpers.batch_iter(y, tx, batch_size = batch_size):
            loss = costs.compute_loss(y, tx, w)
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            ws.append(w)
            losses.append(loss)
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    #raise NotImplementedError
    return losses, ws

# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
import helpers

def compute_loss(y, tx, w, flag=0):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    
    # w = [1, 2].transpose()

    e = y -np.dot(tx, w)

    if(flag == 0):
        loss_result_mse = 0.5 * np.mean(e**2)
        return loss_result_mse
    
    else:
        loss_result_mae = 0.5 * np.mean(np.abs(e))
        return loss_result_mae
    
    #raise NotImplementedError
 
'''
(height, weight, gender) = helpers.load_data(sub_sample=True, add_outlier=False)
(y,tx) = helpers.build_model_data(height, weight)
#print(compute_loss(y, tx, [1,2]))
print("y's shape", np.shape(y))
print("tx' shape", np.shape(tx))
#print(tx)
'''
import numpy as np

def full_tensor_loss(estim):
    # define F from estim
    return np.linalg.norm(F) ** 2

def partial_tensor_loss(estim, ind=3):
    # if ind == 1: j = k
    # elif ind == 2: i = k
    # elif ind == 3: i = j
    return 0
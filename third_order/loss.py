import numpy as np

def full_tensor_loss(F):
    return np.linalg.norm(F) ** 2

def partial_tensor_loss(F, ind=3):
    return 0.
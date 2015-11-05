import numpy as np


def l1_norm(X):
    return np.linalg.norm(X.reshape(len(X)**2,),ord=1)

def l2_norm(X):
    """
    To penalize the total influence provided or received by one guy
    """
    pass

def elastic_net(X, alpha=.5):
    return alpha * l1_norm(X) + (1.-alpha) * sq_frobenius(X)

def frobenius(X):
    return np.linalg.norm(X,ord='fro')

def sq_frobenius(X):
    return np.linalg.norm(X,ord='fro') ** 2
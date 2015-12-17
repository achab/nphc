import numpy as np
from scipy.stats import kendalltau

def rel_err(A_true, A_pred):
    assert A_pred.shape == A_true.shape, "A_pred and A_true should have the same dimensions."
    A_not_zero = A_true != 0
    average = 0.
    average += np.sum(np.abs(A_pred) * (A_true == 0))
    average += np.sum(np.abs(A_true - A_pred)[A_not_zero] / np.abs(A_true)[A_not_zero])
    average /= (A_true.shape[0] ** 2)
    return average

def rank_corr(A_true, A_pred):
    """
    RankCorr is defined as the averaged Kendall's rank correlation coefficient between each row of A_true and A_pred
    """
    return np.mean([kendalltau(x_true, x_pred) for (x_true, x_pred) in zip(A_true, A_pred)])

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
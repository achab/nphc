import numpy as np
from scipy.stats import kendalltau
from .cumulants import get_K_th

def rel_err(A_true, A_pred):
    assert A_pred.shape == A_true.shape, "A_pred and A_true should have the same dimensions."
    A_not_zero = A_true != 0
    average = 0.
    average += np.sum(np.abs(A_pred) * (A_true == 0))
    average += np.sum(np.abs(A_true - A_pred)[A_not_zero] / np.abs(A_true)[A_not_zero])
    average /= A_true.size
    return average

def rank_corr(A_true, A_pred):
    """
    RankCorr is defined as the averaged Kendall's rank correlation coefficient between each row of A_true and A_pred
    """
    return np.mean([kendalltau(x_true, x_pred) for (x_true, x_pred) in zip(A_true, A_pred)])

def l1_norm(X):
    return np.linalg.norm(X.reshape(len(X)**2,),ord=1)

def frob(X):
    return np.linalg.norm(X,ord='fro')

def elastic_net(X, alpha=.5):
    return alpha * l1_norm(X) + (1.-alpha) * sq_frobenius(X)

def frobenius(X):
    return np.linalg.norm(X,ord='fro')

def sq_frobenius(X):
    return np.linalg.norm(X) ** 2

def mse_K(cumul, R):
    K_from_R = get_K_th(cumul.L,cumul.C,R)
    return .5/(cumul.dim**3) * sq_frobenius(cumul.K - K_from_R)

def mse_K_partial(cumul, R):
#    L = cumul.L
#    C = cumul.C
#    K_part = cumul.K_partial
    from math import sqrt
    dim = int(sqrt(R.shape[0]))
    R = R.reshape(dim,dim)
    K_part_from_R = np.dot(R**2,C.T)
    K_part_from_R += 2*np.dot(R*(C-np.dot(R,np.diag(L))),R.T)
    return np.linalg.norm(K_part - K_part_from_R)
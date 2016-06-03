import numpy as np
from scipy.stats import kendalltau
from cumulants import get_K_c_th

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

#def mse_K(cumul, R):
#    d = cumul.dim
#    if R.shape[0] == d**2:
#        R_ = R.reshape(d,d)
#    else:
#        R_ = R.copy()
#    K_from_R = get_K_th(cumul.L,cumul.C,R_)
#    return .5/(cumul.dim**3) * sq_frobenius(cumul.K - K_from_R)

def mse_K_c(cumul, R):
    d = cumul.dim
    if R.shape[0] == d**2:
        R_ = R.reshape(d,d)
    else:
        R_ = R.copy()
    K_c_from_R = get_K_c_th(cumul.L,cumul.C,R_)
    return .5/(cumul.dim**2) * sq_frobenius(cumul.K_c - K_c_from_R)
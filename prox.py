import numpy as np
from numba import autojit
from numpy.linalg import LinAlgError

#@autojit
def nonnegativity(X):
    return np.maximum(X,0)

#@autojit
def stability(X, alpha=0.999):
    U, s, V = np.linalg.svd(X)
    s[s > alpha] = alpha
    return np.dot(U,np.dot(np.diag(s),V))

#@autojit
def orthogonality(X, relaxed=True):
    if relaxed:
        return stability(X, alpha=1.)
    else:
        # another formula exists: X (X^T X)^{-1/2}
        U, _, V = np.linalg.svd(X)
        return np.dot(U,V)

def prox_zero(X, lbd=1.):
    return X

#@autojit
def prox_l1(X, lbd=1.):
    X_abs = np.abs(X)
    return np.sign(X) * (X_abs - lbd) * (X_abs > lbd)

#@autojit
def prox_l2(X, lbd=1.):
    return 1. / (1. + lbd) * X

#@autojit
def prox_enet(X, lbd=1., alpha=.5):
    return alpha * prox_l1(X, lbd=lbd) / (1. + lbd * (1. - alpha))

#@autojit
def prox_frob(X, lbd=1.):
    pass

#@autojit
def prox_sq_frob(X, lbd=1.):
    return 1. / (1. + lbd) * X

#@autojit
def prox_nuclear(X, lbd=1.):
    U, s, V = np.linalg.svd(X)
    s_thres = prox_l1(s, lbd=lbd)
    return np.dot(U,np.dot(np.diag(s_thres),V))


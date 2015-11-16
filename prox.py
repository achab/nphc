import numpy as np
from numba import autojit
from numpy.linalg import LinAlgError

#@autojit
def nonnegativity(X):
    return np.maximum(X,0)

#@autojit
def stability(X, alpha=0.999):
    try:
        U, s, V = np.linalg.svd(X)
        s[s > alpha] = alpha
        return np.dot(U,np.dot(np.diag(s),V))
    except LinAlgError:
        print("le pb vient de stability")

#@autojit
def orthogonality(X, relaxed=True):
    if relaxed:
        try:
            return stability(X, alpha=1.)
        except LinAlgError:
            print("le pb vient de orthogonality")
    else:
        # another formula exists: X (X^T X)^{-1/2}
        U, _, V = np.linalg.svd(X)
        return np.dot(U,V)

#@autojit
def l1(X, lbd=1.):
    X_abs = np.abs(X)
    return np.sign(X) * (X_abs - lbd) * (X_abs > lbd)

#@autojit
def l2(X, lbd=1.):
    pass

#@autojit
def enet(X, lbd=1., alpha=.5):
    return alpha * l1(X, lbd=lbd) + (1. - alpha) * sq_frob(X, lbd=lbd)

#@autojit
def frob(X, lbd=1.):
    pass

#@autojit
def sq_frob(X, lbd=1.):
    return 1. / (1. + lbd) * X

#@autojit
def nuclear(X, lbd=1.):
    U, s, V = np.linalg.svd(X)
    s_thres = l1(s, lbd=lbd)
    return np.dot(U,np.dot(np.diag(s_thres),V))


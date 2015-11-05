import numpy as np

def nonnegativity(X):
    return np.maximum(X,0)

def orthogonality(X):
    U, _, V = np.linalg.svd(X)
    return np.dot(U,V)

def stability(X, alpha=0.99):
    U, S, V = np.linalg.svd(X)
    S[S > alpha] = alpha
    return np.dot(U,np.dot(S,V))

def l1(X, lbd=1.):
    X_abs = np.abs(X)
    return np.sign(X) *(X_abs - lbd) * (X_abs > lbd)

def l2(X, lbd=1.):
    pass

def enet(X, lbd=1., alpha=.5):
    return alpha * l1(X, lbd=lbd) + (1. - alpha) * sq_frob(X, lbd=lbd)

def frob(X, lbd=1.):
    pass

def sq_frob(X, lbd=1.):
    return 1. / (1. + lbd) * X


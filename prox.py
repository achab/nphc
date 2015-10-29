import numpy as np

def prox_nonnegativity(X):
    return np.maximum(X,0)

def prox_orthogonality(X):
    U, _, V = np.linalg.svd(X)
    return np.dot(U,V)

def prox_stability(X, alpha=0.99):
    U, S, V = np.linalg.svd(X)
    S[S > alpha] = alpha
    return np.dot(U,np.dot(S,V))

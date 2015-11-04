import numpy as np
from prox import prox_nonnegativity, prox_orthogonality, prox_stability

def update_X1(prox, X2, AY1, U2, U4, rho=0.1):
    return prox(0.5 * (X2 - U2 + AY1 + U4), rho=rho)

def update_X2(X1, X3, U2, U3):
    return 0.5 * prox_nonnegativity(X1 + U2 + X3 + U3)

def update_X3(X2, U3, alpha=0.99):
    return prox_stability(X2 - U3, alpha=alpha)

def update_X4(Y2B, U5):
    return prox_orthogonality(Y2B + U5)

def update_Y1(X1, Y2, U1, U4, diagA, C):
    return np.dot(np.diag(1. / (1. + diagA ** 2)), np.dot(np.diag(diagA),(X1 - U4)) - Y2 + C - U1)

def update_Y2(X4, Y1, U1, U5, diagD, O, B, C):
    return np.dot(np.dot(O.T,np.dot(np.diag(1. / (1. + diagD ** 2)),O)),np.dot((X4 - U5),B) - Y1 + C - U1)

def update_U1(U1, Y1, Y2, C):
    return U1 + Y1 + Y2 - C

def update_U2(U2, X1, X2):
    return U2 + X1 - X2

def update_U3(U3, X2, X3):
    return U3 + X3 - X2

def update_U4(U4, X1, AY1):
    return U4 + AY1 - X1

def update_U5(U5, X4, Y2B):
    return U5 + Y2B - X4

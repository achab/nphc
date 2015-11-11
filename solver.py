import numpy as np
from utils import empirical_sqrt_mean, empirical_sqrt_cross_corr
from updates import *
from numba import autojit

@autojit
def admm(estim, prox_fun, X1_0, X4_0, rho=0.1, alpha=0.99, maxiter=100):
    """
    ADMM framework to minimize a prox-capable objective over the matrix of kernel norms.
    """

    # compute diagA, diagD, O, B and C
    diagA = empirical_sqrt_mean(estim.lam)
    diagD, O = empirical_sqrt_cross_corr(estim)
    B = np.dot(O.T,np.dot(np.diag(diagD,O)))
    C = np.diag(1. / diagA)

    # initialize parameters
    X1 = X1_0.copy()
    X2 = X1_0.copy()
    X3 = X1_0.copy()
    X4 = X4_0.copy()
    Y1 = np.dot(np.diag(diagA), X1_0)
    Y2 = np.dot(X4_0, B)
    U1 = np.zeros_like(X1_0)
    U2 = np.zeros_like(X1_0)
    U3 = np.zeros_like(X1_0)
    U4 = np.zeros_like(X1_0)
    U5 = np.zeros_like(X1_0)

    for _ in range(maxiter):
        X1[:] = update_X1(prox_fun, X2, Y1, U2, U4, diagA, rho=rho)
        X2[:] = update_X2(X1, X3, U2, U3)
        X3[:] = update_X3(X2, U3, alpha=alpha)
        X4[:] = update_X4(Y2, U5, B)
        Y1[:] = update_Y1(X1, Y2, U1, U4, diagA, C)
        Y2[:] = update_Y2(X4, Y1, U1, U5, diagD, O, B, C)
        U1[:] = update_U1(U1, Y1, Y2, C)
        U2[:] = update_U2(U2, X1, X2)
        U3[:] = update_U3(U3, X2, X3)
        U4[:] = update_U4(U4, X1, Y1, diagA)
        U5[:] = update_U5(U5, X4, Y2, B)

    return X1

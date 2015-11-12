import numpy as np
from admm_hawkes.utils import empirical_sqrt_mean, empirical_sqrt_cross_corr
import admm_hawkes.updates as upd
from admm_hawkes.loss import l1_norm, sq_frobenius
import matplotlib.pyplot as plt
from numba import autojit

@autojit
def admm(estim, prox_fun, X1_0, X4_0, alpha_truth, rho=0.1, alpha=0.99, maxiter=100):
    """
    ADMM framework to minimize a prox-capable objective over the matrix of kernel norms.
    """

    # compute diagA, diagD, O, B and C
    diagA = empirical_sqrt_mean(estim.lam)
    diagD, O = empirical_sqrt_cross_corr(estim)
    B = np.dot(O,np.dot(np.diag(diagD),O.T))
    C = np.diag(1. / diagA)

    # initialize parameters
    X1 = X1_0.copy()
    X2 = X1_0.copy()
    X3 = X1_0.copy()
    X4 = X4_0.copy()
    Y1 = np.dot(np.diag(1. / diagA), X1_0)
    #Y1 = X1_0.copy()
    Y2 = np.dot(X4_0, np.dot(O,np.dot(np.diag(1. / diagD),O.T)))
    #Y2 = X1_0.copy()
    U1 = np.zeros_like(X1_0)
    U2 = np.zeros_like(X1_0)
    U3 = np.zeros_like(X1_0)
    U4 = np.zeros_like(X1_0)
    U5 = np.zeros_like(X1_0)

    loss = []
    error = []
    norm_alpha_truth = sq_frobenius(alpha_truth)

    for _ in range(maxiter):
        X1[:] = upd.update_X1(prox_fun, X2, Y1, U2, U4, diagA, rho=rho)
        assert not np.iscomplex(upd.update_X1(prox_fun, X2, Y1, U2, U4, diagA, rho=rho)).any(), "la valeur complexe vient de X1"
        X2[:] = upd.update_X2(X1, X3, U2, U3)
        assert not np.iscomplex(upd.update_X2(X1, X3, U2, U3)).any(), "la valeur complexe vient de X2"
        X3[:] = upd.update_X3(X2, U3, alpha=alpha)
        assert not np.iscomplex(upd.update_X3(X2, U3, alpha=alpha)).any(), "la valeur complexe vient de X3"
        X4[:] = upd.update_X4(Y2, U5, B)
        assert not np.iscomplex(upd.update_X4(Y2, U5, B)).any(), "la valeur complexe vient de X4"
        Y1[:] = upd.update_Y1(X1, Y2, U1, U4, diagA, C)
        assert not np.iscomplex(upd.update_Y1(X1, Y2, U1, U4, diagA, C)).any(), "la valeur complexe vient de Y1"
        Y2[:] = upd.update_Y2(X4, Y1, U1, U5, diagD, O, B, C)
        assert not np.iscomplex(upd.update_Y2(X4, Y1, U1, U5, diagD, O, B, C)).any(), "la valeur complexe vient de Y2"
        U1[:] = upd.update_U1(U1, Y1, Y2, C)
        assert not np.iscomplex(upd.update_U1(U1, Y1, Y2, C)).any(), "la valeur complexe vient de U1"
        U2[:] = upd.update_U2(U2, X1, X2)
        assert not np.iscomplex(upd.update_U2(U2, X1, X2)).any(), "la valeur complexe vient de U2"
        U3[:] = upd.update_U3(U3, X2, X3)
        assert not np.iscomplex(upd.update_U3(U3, X2, X3)).any(), "la valeur complexe vient de U3"
        U4[:] = upd.update_U4(U4, X1, Y1, diagA)
        assert not np.iscomplex(upd.update_U4(U4, X1, Y1, diagA)).any(), "la valeur complexe vient de U4"
        U5[:] = upd.update_U5(U5, X4, Y2, B)
        assert not np.iscomplex(upd.update_U5(U5, X4, Y2, B)).any(), "la valeur complexe vient de U5"
#        loss.append(objective(X1))
        error.append(sq_frobenius(X1-alpha_truth)/norm_alpha_truth)

    print("||X1 - X_2|| = ", np.linalg.norm(X1-X2))
    print("||X2 - X_3|| = ", np.linalg.norm(X2-X3))
    print("||U1|| = ", np.linalg.norm(U1))
    print("||U2|| = ", np.linalg.norm(U2))
    print("||U3|| = ", np.linalg.norm(U3))
    print("||U4|| = ", np.linalg.norm(U4))
    print("||U5|| = ", np.linalg.norm(U5))

#    plt.figure()
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.plot(loss)
#    plt.plot(error)
#    plt.show()

    return X1

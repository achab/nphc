import numpy as np
from nphc.utils.cumulants import Cumulants, get_K_c_th, get_C_th
from numba import autojit
from nphc.utils.prox import prox_zero
from nphc.utils.metrics import mse_K, rel_err



#######################################################
### This module contains the optimization functions ###
### to minimize MSE on integrated 3^rd cumulant     ###
#######################################################

@autojit
def gradient_f_ijk(cumul,R,i,j,k,alpha=0.):
    return grad_ijk_third_order_norm(cumul,R,i,j,k) + alpha*grad_ij_second_order_norm(cumul,R,i,j)

@autojit
def gradient_f(cumul,R,alpha=0.):
    return grad_part_third_order_norm(cumul,R) + alpha*grad_second_order_norm(cumul,R)

#@autojit
def gradient_g_ij(cumul,R,i,j,alpha=0.):
    return grad_ij_part_third_order_norm(cumul,R,i,j) + alpha*grad_ij_second_order_norm(cumul,R,i,j)

#@autojit
def gradient_g(cumul,R,alpha=0.):
    return grad_part_third_order_norm(cumul,R) + alpha*grad_second_order_norm(cumul,R)

def grad_second_order_norm(cumul,R):
    d = cumul.dim
    C = cumul.C
    L = cumul.L
    C_from_R = get_C_th(L,R)
    diff_C = C_from_R - C
    res = np.einsum('m,im,ij->jm',L,R,diff_C)
    res += np.einsum('m,jm,ij->im',L,R,diff_C)
    return 1./(d**2)*res

def grad_ij_second_order_norm(cumul,R,i,j):
    d = cumul.dim
    C = cumul.C
    L = cumul.L
    grad = np.zeros((d,d))
    grad[j] = R[i] * L
    grad[i] = R[j] * L
    k_ij = np.sum(L * R[i] * R[j])
    return (k_ij - C[i,j]) * grad


#####################################
# a closure to update metrics saved #
#####################################

#@autojit
def inspector(loss_fun, R_true, verbose=False, n_iter=100):
    """A closure called to update metrics after each iteration."""
    objectives = []
    errors = []
    it = [0]  # This is a hack to be able to modify 'it' inside the closure.
    def inspector_cl(Rk):
        obj = loss_fun(Rk)
        err = rel_err(R_true,Rk)
        objectives.append(obj)
        errors.append(err)
        if verbose == True:
            if it[0] == 0:
                print( ' | '.join([name.center(8) for name in ["it", "obj", "norm_grad"]]))
            if it[0] % (n_iter / 5) == 0:
                print(' | '.join([("%d" % it[0]).rjust(8), ("%.2e" % obj).rjust(8), ("%.2e" % err).rjust(8)]))
            it[0] += 1
    inspector_cl.obj = objectives
    inspector_cl.err = errors
    return inspector_cl


###########
# solvers #
###########

@autojit
def gd(R0, grad_fun, n_iter=100, step=1., prox=prox_zero, lbd= 1., callback=None):
    """Basic gradient descent algorithm."""
    R = R0.copy()

    if callback is not None:
        for _ in range(n_iter):
            # Update metrics after each iteration.
            callback(R)
            R -= step * grad_fun(R)
            R[:] = prox(R,lbd)
    else:
        for _ in range(n_iter):
            R -= step * grad_fun(R)
            R[:] = prox(R,lbd)
    return R

@autojit
def sgd(R0, grad_i_fun, n_iter=100, step=1., prox=prox_zero, lbd= 1., callback=None):
    """Stochastic gradient descent algorithm."""
    R = R0.copy()
    d = R.shape[0]

    if callback is not None:
        for n in range(n_iter):
            # Update metrics after each iteration.
            callback(R)
            i,j,k = np.random.randint(d,size=3)
            R -= step * grad_i_fun(R,i,j,k) / (np.sqrt(n + 1))
            R[:] = prox(R,lbd)
    else:
        for n in range(n_iter):
            i,j,k = np.random.randint(d,size=3)
            R -= step * grad_i_fun(R,i,j,k) / (np.sqrt(n + 1))
            R[:] = prox(R,lbd)
    return R

@autojit
def nag(R0, grad_fun, n_iter=100, step=1., prox=prox_zero, lbd= 1., callback=None):
    """Nesterov accelerated gradient algorithm."""
    R = R0.copy()
    R_old = R0.copy()
    Y = np.zeros_like(R0)

    if callback is not None:
        for n in range(n_iter):
            # Update metrics after each iteration.
            callback(R)
            Y[:] = R + (n-2.)/(n+1.)*(R-R_old)
            R_old[:] = R
            Y -= step * grad_fun(Y)
            R[:] = prox(Y,lbd)
    else:
        for n in range(n_iter):
            Y[:] = R + (n-2.)/(n+1.)*(R-R_old)
            R_old[:] = R
            Y -= step * grad_fun(Y)
            R[:] = prox(Y,lbd)
    return R

@autojit
def adagrad(R0, grad_i_fun, n_iter=100, step=1., prox=prox_zero, lbd= 1., callback=None,eps=1e-4):
    """AdaGrad algorithm."""
    R = R0.copy()
    d = R.shape[0]
    diagG = .0001*np.ones((d,d))

    if callback is not None:
        for n in range(n_iter):
            # Update metrics after each iteration.
            callback(R)
            i,j,k = np.random.randint(d,size=3)
            grad = grad_i_fun(R,i,j,k)
            diagG += grad**2
            R -= step * grad / np.sqrt(diagG)
            R[:] = prox(R,lbd)
    else:
        for n in range(n_iter):
            i,j,k = np.random.randint(d,size=3)
            grad = grad_i_fun(R,i,j,k)
            diagG += grad**2
            R -= step * grad / np.sqrt(diagG)
            R[:] = prox(R,lbd)
    return R

#@autojit
def adadelta(R0, grad_i_fun, gamma=.95, eps=1e-6, n_iter=100, prox=prox_zero, lbd= 1., callback=None):
    """AdaDelta algorithm."""
    R = R0.copy()
    d = R.shape[0]
    g_sq = np.zeros((d,d))
    X_sq = np.zeros((d,d))
    dR = np.zeros((d,d))
    def rms(X_):
        return np.sqrt(X_+eps)

    if callback is not None:
        for n in range(n_iter):
            # Update metrics after each iteration.
            callback(R)
            i,j,k = np.random.randint(d,size=3)
            grad = grad_i_fun(R,i,j,k)
            g_sq *= gamma
            g_sq += (1.-gamma)*grad**2
            dR = rms(X_sq)/rms(g_sq)*grad
            X_sq *= gamma
            X_sq += (1.-gamma)*dR**2
            R -= dR
            R[:] = prox(R,lbd)
    else:
        for n in range(n_iter):
            i,j,k = np.random.randint(d,size=3)
            grad = grad_i_fun(R,i,j,k)
            g_sq *= gamma
            g_sq += (1.-gamma)*grad**2
            dR = rms(X_sq)/rms(g_sq)*grad
            X_sq *= gamma
            X_sq += (1.-gamma)*dR**2
            R -= dR
            R[:] = prox(R,lbd)
    return R


#####################
# gradient checking #
#####################

if __name__ == "__main__":

    N = []
    sizes = np.random.randint(low=15,high=20,size=3)
    for n in sizes:
        process = np.sort(np.random.rand(n))
        N.append(process)

    cumul = Cumulants(N)
    cumul.hMax = .2
    cumul.compute_all()

    simple_obj = lambda R: mse_K(cumul,R)
    simple_grad = lambda R: gradient_f(cumul,R)

    d = cumul.dim
    R0 = np.random.rand(d**2).reshape(d,d)
    rand_mat = np.zeros((d,d))
    a,b = np.random.randint(d,size=2)
    rand_mat[a,b] += 1.
    erreur = []
    res1 = np.einsum('ij,ij',simple_grad(R0),rand_mat)
    for u in np.arange(0,-13,-1):
        eps = 10**u
        res2 = (simple_obj(R0+eps*rand_mat)-simple_obj(R0-eps*rand_mat))/(2*eps)
        erreur.append(abs(res1-res2)/abs(res1))
    import matplotlib.pyplot as plt
    plt.plot(np.log(erreur))
    plt.show()

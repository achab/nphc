import numpy as np
from utils.cumulants import Cumulants, get_K_th, get_K_part_th
from numba import autojit
from utils.prox import prox_zero

from utils.metrics import mse_K, rel_err


#######################################################
### This module contains the optimization functions ###
### to minimize MSE on integrated 3^rd cumulant     ###
#######################################################

@autojit
def gradient_f_ijk(cumul,R,i,j,k):
    d = cumul.dim
    C = cumul.C
    L = cumul.L
    K = cumul.K
    grad = np.zeros((d,d))
    grad[i] = R[j]*C[k] + R[k]*C[j] - 2*L*R[j]*R[k]
    grad[j] = R[i]*C[k] + R[k]*C[i] - 2*L*R[i]*R[k]
    grad[k] = R[i]*C[j] + R[j]*C[i] - 2*L*R[i]*R[j]
    k_ijk = np.sum(R[i]*R[j]*C[k] + R[i]*C[j]*R[k] + C[i]*R[j]*R[k] - 2*L*R[i]*R[j]*R[k])
    return (k_ijk - K[i,j,k])*grad

@autojit
def gradient_f(cumul,R):
    d = cumul.dim
    L = cumul.L
    C = cumul.C
    K_from_R = get_K_th(L,C,R)
    diff_K = K_from_R - cumul.K
    res = np.zeros((d,d))
    for idx in ['ijk','kij','jki']:
        res += np.einsum(idx+',im,jm->km',diff_K,R,C)
        res += np.einsum(idx+',im,jm->km',diff_K,C,R)
        res -= 2*np.einsum(idx+',m,im,jm->km',diff_K,L,R,R)
    return 1./(d**3)*res

#@autojit
def gradient_g_ij(cumul,R,i,j):
    d = cumul.dim
    C = cumul.C
    L = cumul.L
    K_part = cumul.K_part
    grad = np.zeros((d,d))
    grad[i] = R[i]*C[j] + R[j]*C[i] - 2*L*R[i]*R[j]
    grad[j] = R[i]*C[i] - 2*L*R[i]**2
    grad *= 2.
    k_iij = np.sum(C[j]*R[i]**2 + 2*R[i]*C[i]*R[j] - 2*L*R[j]*R[i]**2)
    return (k_iij- K_part[i,j])*grad

#@autojit
def gradient_g(cumul,R):
    d = cumul.dim
    L = cumul.L
    C = cumul.C
    K_part_from_R = get_K_part_th(L,C,R)
    diff_K = K_part_from_R - cumul.K_part
    res = np.einsum('ij,im,jm->im',diff_K,R,C)
    res += np.einsum('ij,im,jm->im',diff_K,C,R)
    res -= 2*np.einsum('ij,m,im,jm->im',diff_K,L,R,R)
    res += np.einsum('ij,im->jm',diff_K,R*C)
    res -= np.einsum('ij,m,im->jm',diff_K,L,R**2)
    return 2./(d**2)*res

def grad_second_order_norm():
    pass

def grad_ij_second_order_norm():
    pass

def grad_third_order_norm():
    pass

def grad_ijk_third_order_norm():
    pass

def grad_part_third_order_norm():
    pass

def grad_ij_part_third_order_norm():
    pass

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

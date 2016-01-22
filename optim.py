import numpy as np
from numba import autojit
from .cumulants import Cumulants, get_K_th
from .metrics import mse_K, rel_err
from .prox import prox_zero

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
    dKdR = np.einsum('im,jm->ijm',R,C)
    dKdR += np.einsum('im,jm->ijm',C,R)
    dKdR -= 2*np.einsum('m,im,jm->ijm',L,R,R)
    diff_K = K_from_R - cumul.K
    res = np.einsum('ijk,ijm->km',diff_K,dKdR)
    res += np.einsum('ijk,ikm->jm',diff_K,dKdR)
    res += np.einsum('ijk,jkm->im',diff_K,dKdR)
    return 1./(d**3)*res

#####################################
# a closure to update metrics saved #
#####################################

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

def sgd(R0, grad_i_fun, n_dim=3, n_iter=100, step=1., prox=prox_zero, lbd= 1., callback=None):
    """Stochastic gradient descent algorithm."""
    R = R0.copy()
    d = R.shape[0]

    if callback is not None:
        for n in range(n_iter):
            # Update metrics after each iteration.
            callback(R)
            idx = np.random.randint(d,size=n_dim)
            R -= step * grad_i_fun(R, idx) / (np.sqrt(n + 1))
            R[:] = prox(R,lbd)
    else:
        for n in range(n_iter):
            idx = np.random.randint(d,size=n_dim)
            R -= step * grad_i_fun(R, idx) / (np.sqrt(n + 1))
            R[:] = prox(R,lbd)
    return R

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
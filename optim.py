import numpy as np
from numba import autojit
from cumulants import Cumulants, get_K_th
from metrics import mse_K

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
    grad[i] = R[j]*C[k] + R[k]*C[j] - 2*L_*R[j]*R[k]
    grad[j] = R[i]*C[k] + R[k]*C[i] - 2*L_*R[i]*R[k]
    grad[k] = R[i]*C[j] + R[j]*C[i] - 2*L_*R[i]*R[j]
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

if __name__ == "__main__":

    N = []
    sizes = np.random.randint(low=15,high=20,size=3)
    for n in sizes:
        process = np.sort(np.random.rand(n))
        N.append(process)

    cumul = Cumulants(N)
    cumul.hMax = .2
    cumul.compute_all()

    # Gradient checking
    simple_obj = lambda R: mse_K(cumul,R)
    simple_grad = lambda R: gradient_f(cumul,R)

    d = cumul.dim
    R0 = np.random.rand(d**2).reshape(d,d)
    rand_mat = np.zeros((d,d))
    a,b = np.random.randint(d), np.random.randint(d)
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
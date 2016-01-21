import numpy as np
from .transform import empirical_sqrt_mean, empirical_cross_corr, integrated_claw
from mlpp.hawkesnoparam import Estim
from numba import autojit

class SimpleHawkes:

    def __init__(self, N=None):
        try:
            self.dim = len(N)
        except:
            self.dim = 0
        self.N = N
        self.L = np.empty(self.dim)

    def set_L(self):
        if self.N is not None:
            for i, process in enumerate(self.N):
                self.L[i] = (process[-1] - process[0]) / len(process)


class Cumulants(SimpleHawkes):

    def __init__(self,hMax=40.):
        self.C = None
        self.K = None
        self.K_part = None
        self.hMax = hMax

    def set_C(self):
        self.C = get_C(self,self.L,self.hMax)

    def set_C_th(self, R_truth=None):
        assert R_truth is not None, "You should provide R_truth to compute theoretical terms."
        self.C_th = np.dot(R_truth,np.dot(np.diag(self.L),R_truth.T))

    def set_K(self):
        pass

    def set_K_th(self, R_truth=None):
        assert R_truth is not None, "You should provide R_truth to compute theoretical terms."
        self.K_th = get_K_th(self.L,self.C_th,R_truth)

    def set_K_partial(self):
        self.K_part = get_K_part(self)

@autojit
def get_C(hk,L,H):
    d = len(L)
    A = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            A[i,j] = A_ij(hk,i,j,H)
    return 2*(A - np.einsum('i,j->ij',L,L)*H) + np.diag(L)

def get_C_claw(estim):
    G = integrated_claw(estim, method='gauss')
    np.fill_diagonal(G, G.diagonal()+1)
    C = np.einsum('i,ij->ij', estim.lam, G.T)
    # the following line cancels the edge effects
    C = .5 * (C + C.T)
    return C

def get_K_th(L,C,R):
    d = len(L)
    K = np.zeros((d,d,d))
    K += np.einsum('im,jm,km->ijk',R,R,C)
    K += np.einsum('im,jm,km->ijk',R,C,R)
    K += np.einsum('im,jm,km->ijk',C,R,R)
    K -= 2*np.einsum('m,im,jm,km->ijk',L,R,R,R)
    return K

def get_K_part_th(L,C,R):
    pass

def get_K(hk,L,H):
    pass

def get_K_part(L, C, R):
    K_part = np.zeros((len(L),len(L)))
    K_part -= np.einsum('ii,j->ij',C,L)
    K_part -= 2*np.einsum('ij,i->ij',C,L)
    K_part += 2*np.einsum('i,j->ij',L**2,L)
    return K_part

@autojit
def A_ij(hk,i,j,H):
    res = 0
    u = 0
    count = 0
    T_ = hk.time
    Z_i = hk.get_full_process()[i]
    Z_j = hk.get_full_process()[j]
    n_i = len(Z_i)
    n_j = len(Z_j)
    if H >= 0:
        for tau in Z_i:
            while u < n_j and Z_j[u] < tau:
                u += 1
            v = u
            while v < n_j and Z_j[v] < tau + H:
                v += 1
            if v < n_j:
                count += 1
                res += v-u
    else:
        for tau in Z_i:
            while u < n_j and Z_j[u] <= tau:
                u += 1
            v = u
            while v >= 0 and Z_j[v] > tau + H:
                v -= 1
            if v >= 0:
                count += 1
                res += u-1-v
    res -= (i==j)*count
    if count < n_i:
        res *= n_i * 1. / count
    res /= T_
    return res

@autojit
def B_ijk(hk,i,j,k,H):
    res = 0
    u = 0
    x = 0
    count = 0
    T_ = hk.time
    H_ = abs(H)
    Z_i = hk.get_full_process()[i]
    Z_j = hk.get_full_process()[j]
    Z_k = hk.get_full_process()[k]
    n_i = len(Z_i)
    n_j = len(Z_j)
    n_k = len(Z_k)
    for tau in Z_k:
        # work on Z_i
        while u < n_i and Z_i[u] <= tau:
            u += 1
        v = u
        while v >= 0 and Z_i[v] > tau - H_:
            v -= 1
        # work on Z_j
        while x < n_j and Z_j[x] <= tau:
            x += 1
        y = x
        while y >= 0 and Z_j[y] > tau - H_:
            y -= 1
        # check if this step is admissible
        if y >= 0 and v >= 0:
            count += 1
            res += (u-1-v-(i==k))*(x-1-y-(j==k))
    if count < n_k:
        res *= n_k * 1. / count
    res /= T_
    return res

@autojit
def moment3_ijk(hk,i,j,k,A_,F_,L,H):
    res = 0
    res += (i==j)*(i==k)*L[i]
    res += F_[i,j,k] - (i==j)*A_[k,i]
    res += F_[j,k,i] - (k==j)*A_[i,j]
    res += F_[k,i,j] - (i==k)*A_[j,k]
    res += (i==j)*(A_[k,i]+A_[i,k])
    res += (i==k)*(A_[j,i]+A_[i,j])
    res += (k==j)*(A_[k,i]+A_[i,k])
    return res

@autojit
def moment3(hk,A_,F_,L,H):
    pass
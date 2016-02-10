import numpy as np
from numba import autojit

from tests.int_claw import integrated_claw


class SimpleHawkes(object):

    def __init__(self, N=[], sort_process=False):
        self.dim = len(N)
        if sort_process:
            self.N = []
            for i, process in enumerate(N):
                self.N.append(np.sort(N[i]))
        else:
            self.N = N
        self.L = np.empty(self.dim)
        self.set_L()
        if self.dim > 0:
            self.time = max([x[-1] for x in N])
        else:
            self.time = 0.

    def set_L(self):
        if self.N is not None:
            for i, process in enumerate(self.N):
                self.L[i] = len(process) / (process[-1] - process[0])


class Cumulants(SimpleHawkes):

    def __init__(self,N=[],hMax=40.):
        super().__init__(N)
        self.C = None
        self.C_th = None
        self.K = None
        self.K_th = None
        self.K_part = None
        self.K_part_th = None
        self.R_true = None
        self.hMax = hMax

    @autojit
    def compute_A(self,H=0.):
        if H == 0.:
            hM = -self.hMax
        else:
            hM = H
        d = self.dim
        A = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                A[i,j] = A_ij(self,i,j,hM)
        self.A = A

    @autojit
    def compute_B(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        assert self.hMax > 0, "self.hMax should be nonnegative."
        d = self.dim
        B = np.zeros((d,d,2))
        for i in range(d):
            for j in range(d):
                B[i,j,0] = F_ijk(self,i,i,j,hM)
                B[i,j,1] = F_ijk(self,j,i,i,hM)
        self.B = B

    @autojit
    def compute_F(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        assert self.hMax > 0, "self.hMax should be nonnegative."
        d = self.dim
        F = np.zeros((d,d,d))
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    F[i,j,k] = F_ijk(self,i,j,k,hM)
        self.F = F

    def set_R_true(self,R_true):
        self.R_true = R_true

    def set_C(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
            self.compute_A(hM)
        self.C = get_C(self.A,self.L,hM)

    def set_C_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.C_th = get_C_th(self.L, self.R_true)

    def set_K(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
            self.compute_A(hM)
            self.compute_F(hM)
            self.set_C(hM)
        assert self.C is not None, "You should first set C using the function 'set_C'."
        self.K = get_K(self.A,self.F,self.L,self.C,hM)

    def set_K_th(self):
        assert self.R_true is not None, "You should provide R_true."
        assert self.C_th is not None, "You should provide C_th to compute K_th."
        self.K_th = get_K_th(self.L,self.C_th,self.R_true)

    def set_K_part(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
            self.compute_A(hM)
            self.compute_B(hM)
            self.set_C(hM)
        self.K_part = get_K_part(self.A,self.B,self.L,self.C,hM)

    def set_K_part_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.K_part_th = get_K_part_th(self.L,self.C_th,self.R_true)

    def compute_all(self,H=0.):
        self.compute_A(H)
        self.compute_F(H)
        self.set_C(H)
        self.set_K(H)

    def compute_all_part(self,H=0.):
        self.compute_A(H)
        self.compute_B(H)
        self.set_C(H)
        self.set_K_part(H)



@autojit
def get_C(A,L,H):
    return A+A.T - 2*np.einsum('i,j->ij',L,L)*H + np.diag(L)

@autojit
def get_C_claw(estim):
    G = integrated_claw(estim, method='gauss')
    np.fill_diagonal(G, G.diagonal()+1)
    C = np.einsum('i,ij->ij', estim.lam, G.T)
    # the following line cancels the edge effects
    C = .5 * (C + C.T)
    return C

@autojit
def get_C_th(L, R):
    return np.dot(R,np.dot(np.diag(L),R.T))

@autojit
def get_K(A,F,L,C,H):
    I = np.eye(len(L))
    K1 = F.copy()
    K1 -= np.einsum('jk,ij->ijk',I,A)
    K1 += np.einsum('ij,ik->ijk',I,A+A.T)
    K1 -= 2*np.einsum('i,jk->ijk',L,C)*H
    K = K1.copy()
    K += np.einsum('jki',K1)
    K += np.einsum('kij',K1)
    K += np.einsum('ij,ik,i->ijk',I,I,L)
    K -= 4*np.einsum('i,j,k->ijk',L,L,L)*H**2
    return K

@autojit
def get_K_th(L,C,R):
    d = len(L)
    if R.shape[0] == d**2:
        R_ = R.reshape(d,d)
    else:
        R_ = R.copy()
    K1 = np.einsum('im,jm,km->ijk',C,R_,R_)
    K = K1.copy()
    K += np.einsum('jki',K1)
    K += np.einsum('kij',K1)
    K -= 2*np.einsum('m,im,jm,km->ijk',L,R_,R_,R_)
    return K

@autojit
def get_K_part(A,B,L,C,H):
    K_part = A.copy()
    K_part += np.diag(L)
    K_part += 2*np.diag(np.diag(A))
    K_part += B[:,:,0]
    K_part += 2*B[:,:,1]
    K_part -= 4*H**2*np.einsum('i,j->ij',L**2,L)
    K_part -= 2*H*np.einsum('ii,j->ij',C,L)
    K_part -= 4*H*np.einsum('ij,i->ij',C,L)
    return K_part

@autojit
def get_K_part_th(L,C,R):
    d = len(L)
    if R.shape[0] == d**2:
        R_ = R.reshape(d,d)
    else:
        R_ = R.copy()
    K_part = np.dot(R*R,C.T)
    K_part += 2*np.dot(R_*(C-np.dot(R_,np.diag(L))),R_.T)
    return K_part

@autojit
def A_ij(hk,i,j,H):
    res = 0
    u = 0
    count = 0
    T_ = hk.time
    if isinstance(hk,Cumulants):
        Z_i = hk.N[i]
        Z_j = hk.N[j]
    else:
        Z_i = hk.get_full_process()[i]
        Z_j = hk.get_full_process()[j]
    n_i = len(Z_i)
    n_j = len(Z_j)
    if H >= 0:
        for tau in Z_j:
            while u < n_i and Z_i[u] < tau:
                u += 1
            v = u
            while v < n_i and Z_i[v] < tau + H:
                v += 1
            if v < n_i:
                count += 1
                res += v-u
        res -= (i==j)*count
        if count < n_j:
            res *= n_j * 1. / count
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
def F_ijk(hk,i,j,k,H):
    res = 0
    u = 0
    x = 0
    count = 0
    T_ = hk.time
    H_ = abs(H)
    if isinstance(hk,Cumulants):
        Z_i = hk.N[i]
        Z_j = hk.N[j]
        Z_k = hk.N[k]
    else:
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


#old

@autojit
def moment3_ijk(i,j,k,A_,F_,L,H):
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
def moment3(A_,F_,L,H):
    I = np.eye(len(L))
    M1 = F_.copy()
    M1 -= np.einsum('jk,ij->ijk',I,A_)
    M1 += np.einsum('ij,ik->ijk',I,A_+A_.T)
    M = M1.copy()
    M += np.einsum('jki',M1)
    M += np.einsum('kij',M1)
    M += np.einsum('ij,ik,i->ijk',I,I,L)
    return M

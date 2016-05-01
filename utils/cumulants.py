import numpy as np
from numba import autojit


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
        self.time = max([x[-1] for x in N]) * (self.dim > 0)
        self.set_L()

    def set_L(self):
        if self.dim > 0:
            for i, process in enumerate(self.N):
                self.L[i] = len(process) / self.time
                #self.L[i] = len(process) / (process[-1] - process[0])


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
        self.H = None

    @autojit
    def set_B(self,H=0.):
        if H == 0.:
            hM = -self.hMax
        else:
            hM = abs(H)
        d = self.dim
        self.B = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                self.B[i,j] = A_ij(self,i,j,-hM,0)

    @autojit
    def set_E(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        self.E = np.zeros((d,d,d))
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    self.E[i,j,k] = E_ijk(self,i,j,k,hM)

    @autojit
    def set_M(self, H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        assert self.C is not None, "You should first set C using the function 'set_C'."
        d = self.dim
        self.M = np.zeros((d,d,d))
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    self.M[i,j,k] = self.L[k] * ( hM * self.C[i,j] - 2 * I_ij(self,i,j,hM) )

    @autojit
    def set_E_c(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        E_c = np.zeros((d,d,2))
        for i in range(d):
            for j in range(d):
                E_c[i,j,0] = E_ijk(self,i,i,j,hM)
                E_c[i,j,1] = E_ijk(self,j,i,i,hM)
        self.E_c = E_c

    def set_M_c(self,H=0.):
        pass

    @autojit
    def set_H(self,method=0,N=1000):
        """
        Set the matrix parameter self.H using different heuristics.
        Method 0 simply set the same H for each couple (i,j).
        Method 1 set the H that minimizes 1/H \int_0^H u c_{ij} (u) du.
        """
        d = self.dim
        if method == 0:
            self.H = self.hMax * np.ones((d,d))
        if method == 1:
            self.H = np.empty((d,d))
            for i in range(d):
                for j in range(d):
                    range_h = np.logspace(-3,3,N)
                    res = []
                    for h in range_h:
                        val = I_ij(self,i,j,h) / h
                        res.append(val)
                    res = np.array(res)
                    self.H[i,j] = range_h[np.argmin(res)]


    def set_R_true(self,R_true):
        self.R_true = R_true

    @autojit
    def set_C(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        self.C = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                self.C[i,j] = A_ij(self,i,j,-hM,hM)
        # we keep the symmetric part to remove edge effects
        self.C[:] = 0.5*(self.C + self.C.T)

    @autojit
    def set_K(self):
        assert self.B is not None, "You should first set B using the function 'set_B'."
        assert self.E is not None, "You should first set E using the function 'set_E'."
        assert self.M is not None, "You should first set M using the function 'set_M'."
        self.K = get_K(self.B,self.E,self.M,self.L)

    def set_K_part(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
            self.set_B(hM)
            self.set_E_c(hM)
            self.set_C(hM)
        self.K_part = get_K_part(self.B,self.E_c,self.L,self.C,hM)

    def set_C_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.C_th = get_C_th(self.L, self.R_true)

    def set_K_th(self):
        assert self.R_true is not None, "You should provide R_true."
        assert self.C_th is not None, "You should provide C_th to set_ K_th."
        self.K_th = get_K_th(self.L,self.C_th,self.R_true)

    def set_K_part_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.K_part_th = get_K_part_th(self.L,self.C_th,self.R_true)

    def set_all(self,H=0.):
        self.set_B(-H)
        self.set_E(H)
        self.set_M(H)
        self.set_C(H)
        self.set_K(H)

    def set_all_part(self,H=0.):
        self.set_B(-H)
        self.set_E_c(H)
        self.set_M_c(H)
        self.set_C(H)
        self.set_K_part(H)


###########
## Empirical cumulants with formula from the paper
###########
@autojit
def get_K(B,E,M,L):
    I = np.eye(len(L))
    K1 = E-M
    K1 += np.einsum('ij,jk->ijk',I,B)
    K = K1.copy()
    K += np.einsum('jki',K1)
    K += np.einsum('kij',K1)
    K += np.einsum('ij,ik,i->ijk',I,I,L)
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


##########
## Theoretical cumulants C, K, K_part
##########
@autojit
def get_C_th(L, R):
    return np.dot(R,np.dot(np.diag(L),R.T))

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
def get_K_part_th(L,C,R):
    d = len(L)
    if R.shape[0] == d**2:
        R_ = R.reshape(d,d)
    else:
        R_ = R.copy()
    K_part = np.dot(R*R,C.T)
    K_part += 2*np.dot(R_*(C-np.dot(R_,np.diag(L))),R_.T)
    return K_part


##########
## Useful fonctions to set_ empirical integrated cumulants
##########
@autojit
def A_ij(cumul,i,j,a,b):
    """

    Computes the mean centered number of jumps of N^j between \tau + a and \tau + b, that is

    \frac{1}{T} \sum_{\tau \in Z^i} ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j (b - a) )

    """
    res = 0
    u = 0
    count = 0
    T_ = cumul.time
    Z_i = cumul.N[i]
    Z_j = cumul.N[j]
    n_i = len(Z_i)
    n_j = len(Z_j)
    L_i = cumul.L[i]
    L_j = cumul.L[j]
    assert a < b, "You should provide a and b such that a < b."
    for tau in Z_i:
        while u < n_j and Z_j[u] <= tau + a:
            u += 1
        v = u
        while v < n_j and Z_j[v] < tau + b:
            v += 1
        if v < n_j and u > 0:
            count += 1
            res += v-u
    if count < n_i and count > 0:
        res *= n_i * 1. / count
    res /= T_
    res -= (b - a) * L_i * L_j
    return res

@autojit
def E_ijk(cumul,i,j,k,H):
    """

    Computes the mean of the centered product of i's and j's jumps between \tau + a and \tau + b, that is

    \frac{1}{T} \sum_{\tau \in Z^k} ( N^i_{\tau} - N^i_{\tau - H} - \delta^{ik} - \Lambda^i H )
                                  * ( N^j_{\tau} - N^j_{\tau - H} - \delta^{jk} - \Lambda^j H )

    """
    res = 0
    u = 0
    x = 0
    count = 0
    T_ = cumul.time
    H_ = abs(H)
    Z_i = cumul.N[i]
    Z_j = cumul.N[j]
    Z_k = cumul.N[k]
    n_i = len(Z_i)
    n_j = len(Z_j)
    n_k = len(Z_k)
    L_i = cumul.L[i]
    L_j = cumul.L[j]
    for tau in Z_k:
        # work on Z_i
        while u < n_i and Z_i[u] <= tau - H_:
            u += 1
        v = u
        while v < n_i and Z_i[v] < tau:
            v += 1
        # work on Z_j
        while x < n_j and Z_j[x] <= tau - H_:
            x += 1
        y = x
        while y < n_j and Z_j[y] < tau:
            y += 1
        # check if this step is admissible
        if y < n_j and x > 0 and v < n_i and u > 0:
            count += 1
            res += (v-u-L_i*H_)*(y-x-L_j*H_)
    if count < n_k and count > 0:
        res *= n_k * 1. / count
    res /= T_
    return res


@autojit
def I_ij(cumul,i,j,H):
    """

    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals

    \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^i \Lambda^j

    """
    res = 0
    u = 0
    count = 0
    T_ = cumul.time
    H_ = abs(H)
    Z_i = cumul.N[i]
    Z_j = cumul.N[j]
    n_i = len(Z_i)
    n_j = len(Z_j)
    L_i = cumul.L[i]
    L_j = cumul.L[j]
    for tau in Z_i:
        while u < n_j and Z_j[u] <= tau - H_:
            u += 1
        v = u
        while v < n_j and Z_j[v] < tau:
            res += tau - Z_j[v]
            count += 1
            v += 1
    if count < n_i and count > 0:
        res *= n_i * 1. / count
    res /= T_
    res -= .5 * (H_**2) * L_i * L_j
    return res


if __name__ == "__main__":
    N = [np.sort(np.random.randint(0,100,size=20)),np.sort(np.random.randint(0,100,size=20))]
    cumul = Cumulants(N,hMax=10)
    cumul.set_B()
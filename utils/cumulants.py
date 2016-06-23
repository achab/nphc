import numpy as np
from numba import autojit, jit, double, int32, int64, float64
from scipy.linalg import inv, pinv, eigh
from tensorflow import Session
from joblib import Parallel, delayed


class Cumulants(object):

    def __init__(self,N=[],hMax=40.,sort_process=False):
        self.dim = len(N)
        if sort_process:
            self.N = []
            for i, process in enumerate(N):
                self.N.append(np.sort(N[i]))
        else:
            self.N = N
        self.L = np.empty(self.dim)
        self.time = max([x[-1]-x[0] for x in N if x is not None and len(x) > 0]) * (self.dim > 0)
        self.set_L()
        self.C = None
        self.C_th = None
        self.K_c = None
        self.K_c_th = None
        self.R_true = None
        self.hMax = hMax
        self.H = None
        # Following attributes are related to weighting matrix in GMM
        self.W_2 = np.ones((self.dim,self.dim))
        self.W_3 = np.ones((self.dim,self.dim))
        self.L_list = []
        self.C_list = []
        self.K_c_list = []

    def set_L(self):
        if self.dim > 0:
            for i, process in enumerate(self.N):
                if process is None:
                    self.L[i] = 0
                else:
                    self.L[i] = len(process) / self.time


    #########
    ## Functions to compute third order cumulant
    #########

    @autojit
    def set_F_c(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        self.F_c = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                self.F_c[i,j] = 2 * ( E_ijk(self.N[j],self.N[i],self.N[j],-hM,hM,-hM,hM,self.time,self.L[j],self.L[i],self.L[j]) - self.L[j]*(2*hM*A_ij(self.N[i],self.N[j],-2*hM,2*hM,self.time,self.L[i],self.L[j]) - 2*I_ij(self.N[j],self.N[i],2*hM,self.time,self.L[j],self.L[i]) ) )
                self.F_c[i,j] += E_ijk(self.N[j],self.N[j],self.N[i],-hM,hM,-hM,hM,self.time,self.L[j],self.L[j],self.L[i]) - self.L[i]*(2*hM*A_ij(self.N[j],self.N[j],-2*hM,2*hM,self.time,self.L[j],self.L[j])  - 2*I_ij(self.N[j],self.N[j],2*hM,self.time,self.L[j],self.L[j]))
        self.F_c /= 3

    #@autojit
    def set_C(self,H=0.,method='parallel'):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        if method == 'classic':
            self.C = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    self.C[i,j] = A_ij(self.N[i],self.N[j],-hM,hM,self.time,self.L[i],self.L[j])
        elif method == 'parallel':
            l = Parallel(-1)(delayed(A_ij)(self.N[i],self.N[j],-hM,hM,self.time,self.L[i],self.L[j]) for i in range(d) for j in range(d))
            self.C = np.array(l).reshape(d,d)
        # we keep the symmetric part to remove edge effects
        self.C[:] = 0.5*(self.C + self.C.T)

    #@autojit
    def set_J(self, H=0.,method='parallel'):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        if method == 'classic':
            self.J = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    self.J[i,j] = I_ij(self.N[i],self.N[j],hM,self.time,self.L[i],self.L[j])
        elif method == 'parallel':
            l = Parallel(-1)(delayed(I_ij)(self.N[i],self.N[j],hM,self.time,self.L[i],self.L[j]) for i in range(d) for j in range(d) )
            self.J = np.array(l).reshape(d,d)
        # we keep the symmetric part to remove edge effects
        self.J[:] = 0.5 * (self.J + self.J.T)

    #@autojit
    def set_E_c(self,H=0.,method='parallel'):
        if H == 0.:
            hM = .5*self.hMax
        else:
            hM = .5*H
        d = self.dim
        if method == 'classic':
            self.E_c = np.zeros((d,d,2))
            for i in range(d):
                for j in range(d):
                    self.E_c[i,j,0] = E_ijk(self.N[i],self.N[j],self.N[j],-hM,hM,-hM,hM,self.time,self.L[i],self.L[j],self.L[j])
                    self.E_c[i,j,1] = E_ijk(self.N[j],self.N[j],self.N[i],-hM,hM,-hM,hM,self.time,self.L[j],self.L[j],self.L[i])
        elif method == 'parallel':
            l1 = Parallel(-1)(delayed(E_ijk)(self.N[i],self.N[j],self.N[j],-hM,hM,-hM,hM,self.time,self.L[i],self.L[j],self.L[j]) for i in range(d) for j in range(d))
            l2 = Parallel(-1)(delayed(E_ijk)(self.N[j],self.N[j],self.N[i],-hM,hM,-hM,hM,self.time,self.L[j],self.L[j],self.L[i]) for i in range(d) for j in range(d))
            self.E_c = np.zeros((d,d,2))
            self.E_c[:,:,0] = np.array(l1).reshape(d,d)
            self.E_c[:,:,1] = np.array(l2).reshape(d,d)

    #@autojit
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
                        val = I_ij(self.N[i],self.N[j],h,self.time,self.L[i],self.L[j]) / h
                        res.append(val)
                    res = np.array(res)
                    self.H[i,j] = range_h[np.argmin(res)]

    def set_K_c(self,H=0.):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        assert self.C is not None, "You should first set C using the function 'set_C'."
        assert self.E_c is not None, "You should first set E using the function 'set_E_c'."
        assert self.J is not None, "You should first set J using the function 'set_J'."
        self.K_c = get_K_c(self.L,self.C,self.J,self.E_c,hM)

    def set_R_true(self,R_true):
        self.R_true = R_true

    def set_C_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.C_th = get_C_th(self.L, self.R_true)

    def set_K_c_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.K_c_th = get_K_c_th(self.L,self.C_th,self.R_true)

    def set_all(self,H=0.):
        print("Starting computation of integrated cumulants...")
        self.set_C(H)
        print("cumul.C is computed !")
        self.set_E_c(H)
        print("cumul.E_c is computed !")
        self.set_J(H)
        print("cumul.J is computed !")
        self.set_K_c(H)
        print("cumul.K_c is computed !")
        if self.R_true is not None:
            self.set_C_th()
            print("cumul.C_th is computed !")
            self.set_K_c_th()
            print("cumul.K_c_th is computed !")
        print("All cumulants are computed !")


    ###########
    ## Functions to compute weighting matrix in GMM
    ###########

    def set_W_2(self, R, weight='diag'):
        R = Session().run(R)
        assert len(self.L_list)*len(self.C_list) > 0, "You should first fill self.L_list and self.C_list"
        assert len(self.L_list) == len(self.C_list), "The lists self.L_list and self.C_list should have the same number of elements."
        if weight == 'diag':
            res = np.zeros_like(R)
            for L, C in zip(self.L_list, self.C_list):
                X = np.dot(R, np.dot(np.diag(L), R.T)) - C
                res += X ** 2
            res[res < 1e-15] = 0.
            self.W_2 = len(self.L_list) * 1./res
        elif weight == 'dense':
            d = R.shape[0]
            res = np.zeros((d**2,d**2))
            for L, C in zip(self.L_list, self.C_list):
                X = np.dot(R, np.dot(np.diag(L), R.T)) - C
                X = X.reshape(R.shape[0]**2,1)
                res += np.outer(X,X)
            res *= 1./len(self.L_list)
            self.W_2 = pinv(res)
        else:
            self.W_2 = np.ones_like(R)

    def set_W_3(self, R, weight='diag'):
        R = Session().run(R)
        assert len(self.L_list)*len(self.C_list)*len(self.K_c_list) > 0, "You should first fill self.L_list, self.C_list and self.K_c_list"
        assert len(self.L_list) == len(self.C_list) == len(self.K_c_list), "The lists self.L_list, self.C_list and self.K_c_list should have the same number of elements."
        if weight == 'diag':
            res = np.zeros_like(R)
            for L, C, K_c in zip(self.L_list, self.C_list, self.K_c_list):
                X = np.dot(R**2,C.T) + 2*np.dot(R*C,R.T) - 2*np.dot( R**2, np.dot(np.diag(L), R.T) ) - K_c
                res += X ** 2
            res[res < 1e-15] = 0.
            self.W_3 = len(self.L_list) * 1./res
        elif weight == 'dense':
            d = R.shape[0]
            res = np.zeros((d**2,d**2))
            for L, C, K_c in zip(self.L_list, self.C_list, self.K_c_list):
                X = np.dot(R**2,C.T) + 2*np.dot(R*C,R.T) - 2*np.dot( R**2, np.dot(np.diag(L), R.T) ) - K_c
                X = X.reshape(R.shape[0]**2,1)
                res += np.outer(X,X)
            res *= 1./len(self.L_list)
            self.W_2 = pinv(res)
        else:
            self.W_3 = np.ones_like(R)


###########
## Empirical cumulants with formula from the paper
###########

@autojit
def get_K_c(L,C,J,E_c,H):
    K_c = np.zeros_like(C)
    K_c += 2*E_c[:,:,0]
    K_c -= np.einsum('j,ij->ij',L,H*C-2*J)
    K_c += E_c[:,:,1]
    K_c -= np.einsum('i,jj->ij',L,H*C-2*J)
    K_c /= 3.
    return K_c

##########
## Theoretical cumulants C, K, K_c
##########

@autojit
def get_C_th(L, R):
    return np.dot(R,np.dot(np.diag(L),R.T))

@autojit
def get_K_c_th(L,C,R):
    d = len(L)
    if R.shape[0] == d**2:
        R_ = R.reshape(d,d)
    else:
        R_ = R.copy()
    K_c = np.dot(R*R,C.T)
    K_c += 2*np.dot(R_*(C-np.dot(R_,np.diag(L))),R_.T)
    return K_c


##########
## Useful fonctions to set_ empirical integrated cumulants
##########
#@jit(double(double[:],double[:],int32,int32,double,double,double), nogil=True, nopython=True)
#@jit(float64(float64[:],float64[:],int64,int64,int64,float64,float64), nogil=True, nopython=True)
@autojit
def A_ij(Z_i,Z_j,a,b,T,L_i,L_j):
    """

    Computes the mean centered number of jumps of N^j between \tau + a and \tau + b, that is

    \frac{1}{T} \sum_{\tau \in Z^i} ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j (b - a) )

    """
    res = 0
    u = 0
    count = 0
    n_i = Z_i.shape[0]
    n_j = Z_j.shape[0]
    trend_j = L_j*(b-a)
    for t in range(n_i):
        tau = Z_i[t]
        if tau + a < 0: continue
        while u < n_j:
            if Z_j[u] <= tau + a:
                u += 1
            else:
                break
        if u == n_j: continue
        v = u
        while v < n_j:
            if Z_j[v] < tau + b:
                v += 1
            else:
                break
        res += v-u-trend_j
        #if v < n_j:
        #    if u > 0:
        #        count += 1
        #        res += v-u
    #if count < n_i:
    #    if count > 0:
    #        res *= n_i * 1. / count
    res /= T
    return res

@autojit
def E_ijk(Z_i,Z_j,Z_k,a_i,b_i,a_j,b_j,T,L_i,L_j,L_k):
    """

    Computes the mean of the centered product of i's and j's jumps between \tau + a and \tau + b, that is

    \frac{1}{T} \sum_{\tau \in Z^k} ( N^i_{\tau + b_i} - N^i_{\tau + a_i} - \Lambda^i * ( b_i - a_i ) )
                                  * ( N^j_{\tau + b_j} - N^j_{\tau + a_j} - \Lambda^j * ( b_j - a_j ) )

    """
    res = 0
    u = 0
    x = 0
    count = 0
    n_i = Z_i.shape[0]
    n_j = Z_j.shape[0]
    n_k = Z_k.shape[0]
    trend_i = L_i*(b_i-a_i)
    trend_j = L_j*(b_j-a_j)
    for t in range(n_k):
        tau = Z_k[t]
        if tau + a_i < 0: continue
        # work on Z_i
        while u < n_i:
            if Z_i[u] <= tau + a_i:
                u += 1
            else:
                break
        v = u
        while v < n_i:
            if Z_i[v] < tau + b_i:
                v += 1
            else:
                break
        # work on Z_j
        while x < n_j:
            if Z_j[x] <= tau + a_j:
                x += 1
            else:
                break
        y = x
        while y < n_j:
            if Z_j[y] < tau + b_j:
                y += 1
            else:
                break
        res += (v-u-trend_i) * (y-x-trend_j)
        # check if this step is admissible
        #if y < n_j and x > 0 and v < n_i and u > 0:
        #    count += 1
        #    res += (v-u-trend_i) * (y-x-trend_j)
    #if count < n_k and count > 0:
    #    res *= n_k * 1. / count
    res /= T
    return res

@autojit
def I_ij(Z_i,Z_j,H,T,L_i,L_j):
    """

    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals

    \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^i \Lambda^j

    """
    n_i = Z_i.shape[0]
    n_j = Z_j.shape[0]
    res = 0
    u = 0
    count = 0
    trend_j = .5 * (H**2) * L_j
    for t in range(n_i):
        tau = Z_i[t]
        tau_minus_H = tau - H
        if tau_minus_H  < 0: continue
        while u < n_j:
            if Z_j[u] <= tau_minus_H :
                u += 1
            else:
                break
        v = u
        while v < n_j:
            tau_minus_tau_p = tau - Z_j[v]
            if tau_minus_tau_p > 0:
                res += tau_minus_tau_p-trend_j
                count += 1
                v += 1
            else:
                break
    #if count < n_i and count > 0:
    #    res *= n_i * 1. / count
    res /= T
    return res

@autojit
def J_ijk(Z_i,Z_j,Z_k,H_i,H_j,T,L_i,L_j,L_k):

    u = 0
    x = 0
    res = 0
    count = 0
    n_i = Z_i.shape[0]
    n_j = Z_j.shape[0]
    n_k = Z_k.shape[0]
    for t in range(n_k):
        res_i = 0
        res_j = 0
        tau = Z_k[t]
        tau_minus_H_i = tau - H_i
        tau_plus_H_i = tau + H_i
        if tau_minus_H_i < 0: continue

        # work on Z_i
        while u < n_i:
            if Z_i[u] <= tau_minus_H_i:
                u += 1
            else:
                break
        v = u
        while v < n_i:
            tau_plus_H_i_minus_tau_i = tau_plus_H_i - Z_i[v]
            if tau_plus_H_i_minus_tau_i > 0:
                res_i += tau_plus_H_i_minus_tau_i
                v += 1
            else:
                break

        # work on Z_j
        tau_minus_H_j = tau - H_j
        tau_plus_H_j = tau + H_j
        while x < n_j:
            if Z_j[x] <= tau_minus_H_j:
                x += 1
            else:
                break
        y = x
        while y < n_j:
            tau_plus_H_j_minus_tau_j = tau_plus_H_j - Z_j[y]
            if tau_plus_H_j_minus_tau_j > 0:
                res_j += tau_plus_H_j_minus_tau_j
                y += 1
            else:
                break

        # check if this step is admissible
        if y < n_j and x > 0 and v < n_i and u > 0:
            count += 1
            res += res_i * res_j

    if count < n_k and count > 0:
        res *= n_k * 1. / count
    res /= T
    res -= 4 * H_i*H_j*L_i*L_j
    return res


if __name__ == "__main__":
    import gzip, pickle
    filename = '/data/users/achab/nphc/datasets/rect/rect_d10_nonsym_2_log10T8_with_params_094.pkl.gz'
    f = gzip.open(filename)
    cumul = pickle.load(f)[0]
    f.close()
    process = cumul.N
    cumul = Cumulants(process,hMax=10)
    cumul.set_all()

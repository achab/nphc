from numba import autojit, jit, double, int32, int64, float64
from scipy.linalg import inv, pinv, eigh
from joblib import Parallel, delayed
from tensorflow import Session
import numpy as np



class Cumulants(object):

    def __init__(self,N=[],hMax=100.):
        self.N = N
        self.N_is_list_of_multivariate_processes = all(isinstance(x, list) for x in self.N)
        if self.N_is_list_of_multivariate_processes:
            self.dim = len(self.N[0])
        else:
            self.dim = len(self.N)
        self.L = np.zeros(self.dim)
        if self.N_is_list_of_multivariate_processes:
            self.time = max([max([x[-1]-x[0] for x in multivar_process if x is not None and len(x) > 0]) for multivar_process in self.N])
        else:
            self.time = max([x[-1]-x[0] for x in self.N if x is not None and len(x) > 0])
        self.C = None
        self.L_th = None
        self.C_th = None
        self.K_c = None
        self.K_c_th = None
        self.R_true = None
        self.mu_true = None
        self.hMax = hMax

    ###########
    ## Decorator to compute the cumulants on each day, and average
    ###########

    def average_if_list_of_multivariate_processes(func):
        def average_cumulants(self,*args,**kwargs):
            if self.N_is_list_of_multivariate_processes:
                if 'parallel' in args:
                    for n, multivar_process in enumerate(self.N):
                        cumul = Cumulants(N=multivar_process)
                        res_one_process = func(cumul,*args,**kwargs)
                        if n == 0:
                            res = np.zeros_like(res_one_process)
                        res += res_one_process
                    res /= n+1
                else:
                    def worker(multivar_process):
                        cumul = Cumulants(N=multivar_process)
                        res_one_process = func(cumul,*args,**kwargs)
                        return res_one_process
                    l = Parallel(-1)(delayed(worker)(m_p) for m_p in self.N)
                    res = np.average(l,axis=0)
            else:
                res = func(self,*args,**kwargs)
            return res
        return average_cumulants
    #########
    ## Functions to compute third order cumulant
    #########

    @average_if_list_of_multivariate_processes
    def compute_L(self):
        self.dim = len(self.N)
        L = np.zeros(self.dim)
        for i, process in enumerate(self.N):
            if process is None:
                L[i] = -1.
            else:
                L[i] = len(process) / self.time
        return L

    @average_if_list_of_multivariate_processes
    def compute_C(self,H=0.,method='parallel'):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        if method == 'classic':
            C = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    C[i,j] = A_ij(self.N[i],self.N[j],-hM,hM,self.time,self.L[j])
        elif method == 'parallel':
            l = Parallel(-1)(delayed(A_ij)(self.N[i],self.N[j],-hM,hM,self.time,self.L[j]) for i in range(d) for j in range(d))
            C = np.array(l).reshape(d,d)
        # we keep the symmetric part to remove edge effects
        C[:] = 0.5 * (C + C.T)
        return C

    @average_if_list_of_multivariate_processes
    def compute_J(self, H=0.,method='parallel'):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        if method == 'classic':
            J = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    J[i,j] = I_ij(self.N[i],self.N[j],hM,self.time,self.L[j])
        elif method == 'parallel':
            l = Parallel(-1)(delayed(I_ij)(self.N[i],self.N[j],hM,self.time,self.L[j]) for i in range(d) for j in range(d) )
            J = np.array(l).reshape(d,d)
        # we keep the symmetric part to remove edge effects
        J[:] = 0.5 * (J + J.T)
        return J

    @average_if_list_of_multivariate_processes
    def compute_E_c(self,H=0.,method='parallel'):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        if method == 'classic':
            E_c = np.zeros((d,d,2))
            for i in range(d):
                for j in range(d):
                    E_c[i,j,0] = E_ijk(self.N[i],self.N[j],self.N[j],-hM,hM,self.time,self.L[i],self.L[j])
                    E_c[i,j,1] = E_ijk(self.N[j],self.N[j],self.N[i],-hM,hM,self.time,self.L[j],self.L[j])
        elif method == 'parallel':
            l1 = Parallel(-1)(delayed(E_ijk)(self.N[i],self.N[j],self.N[j],-hM,hM,self.time,self.L[i],self.L[j]) for i in range(d) for j in range(d))
            l2 = Parallel(-1)(delayed(E_ijk)(self.N[j],self.N[j],self.N[i],-hM,hM,self.time,self.L[j],self.L[j]) for i in range(d) for j in range(d))
            E_c = np.zeros((d,d,2))
            E_c[:,:,0] = np.array(l1).reshape(d,d)
            E_c[:,:,1] = np.array(l2).reshape(d,d)
        return E_c

    def set_L(self):
        self.L = self.compute_L()

    def set_C(self,H=0.,method='parallel'):
        self.C = self.compute_C(H,method)

    def set_J(self, H=0.,method='parallel'):
        self.J = self.compute_J(H,method)

    def set_E_c(self, H=0., method='parallel'):
        self.E_c = self.compute_E_c(H,method)

    def set_K_c(self,H=0.):
        assert self.E_c is not None, "You should first set E using the function 'set_E_c'."
        self.K_c = get_K_c(self.E_c)

    def set_R_true(self,R_true):
        self.R_true = R_true

    def set_mu_true(self,mu_true):
        self.mu_true = mu_true

    def set_L_th(self):
        assert self.R_true is not None, "You should provide R_true."
        assert self.mu_true is not None, "You should provide mu_true."
        self.L_th = get_L_th(self.mu_true, self.R_true)

    def set_C_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.C_th = get_C_th(self.L_th, self.R_true)

    def set_K_c_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.K_c_th = get_K_c_th(self.L_th,self.C_th,self.R_true)

    def set_all(self,H=0.,method="parallel"):
        self.set_L()
        print("L is computed")
        self.set_C(H,method)
        print("C is computed")
        self.set_E_c(H,method)
        self.set_K_c()
        print("K_c is computed")
        if self.R_true is not None and self.mu_true is not None:
            self.set_L_th()
            self.set_C_th()
            self.set_K_c_th()


###########
## Empirical cumulants with formula from the paper
###########

@autojit
def get_K_c(E_c):
    K_c = np.zeros_like(E_c[:,:,0])
    K_c += 2*E_c[:,:,0]
    K_c += E_c[:,:,1]
    K_c /= 3.
    return K_c

##########
## Theoretical cumulants L, C, K, K_c
##########

@autojit
def get_L_th(mu, R):
    return np.dot(R,mu)

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
    K_c = np.dot(C,(R_*R_).T)
    K_c += 2*np.dot(R_,(R_*C).T)
    K_c -= 2*np.dot(np.dot(R_,np.diag(L)),(R_*R_).T)
    return K_c


##########
## Useful fonctions to set_ empirical integrated cumulants
##########
#@jit(double(double[:],double[:],int32,int32,double,double,double), nogil=True, nopython=True)
#@jit(float64(float64[:],float64[:],int64,int64,int64,float64,float64), nogil=True, nopython=True)
@autojit
def A_ij(Z_i,Z_j,a,b,T,L_j):
    """
    Computes the mean centered number of jumps of N^j between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^i} ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j (b - a) )
    """
    res = 0
    u = 0
    n_i = Z_i.shape[0]
    n_j = Z_j.shape[0]
    trend_j = L_j*(b-a)
    for t in range(n_i):
        # count the number of jumps
        tau = Z_i[t]
        if tau + a < 0: continue
        while u < n_j:
            if Z_j[u] <= tau + a:
                u += 1
            else:
                break
        v = u
        while v < n_j:
            if Z_j[v] < tau + b:
                v += 1
            else:
                break
        if v == n_j: continue
        res += v-u-trend_j
    res /= T
    return res

@autojit
def E_ijk(Z_i,Z_j,Z_k,a,b,T,L_i,L_j):
    """
    Computes the mean of the centered product of i's and j's jumps between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^k} ( N^i_{\tau + b} - N^i_{\tau + a} - \Lambda^i * ( b - a ) )
                                  * ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j * ( b - a ) )
    """
    res = 0
    u = 0
    x = 0
    count = 0
    n_i = Z_i.shape[0]
    n_j = Z_j.shape[0]
    n_k = Z_k.shape[0]
    trend_i = L_i*(b-a)
    trend_j = L_j*(b-a)
    C = .5*(A_ij(Z_i,Z_j,-(b-a),b-a,T,L_j)+A_ij(Z_j,Z_i,-(b-a),b-a,T,L_i))
    J = .5*(I_ij(Z_i,Z_j,b-a,T,L_j)+I_ij(Z_j,Z_i,b-a,T,L_i))
    for t in range(n_k):
        tau = Z_k[t]
        if tau + a < 0: continue
        # work on Z_i
        while u < n_i:
            if Z_i[u] <= tau + a:
                u += 1
            else:
                break
        v = u
        while v < n_i:
            if Z_i[v] < tau + b:
                v += 1
            else:
                break
        if v == n_i: continue
        # work on Z_j
        while x < n_j:
            if Z_j[x] <= tau + a:
                x += 1
            else:
                break
        y = x
        while y < n_j:
            if Z_j[y] < tau + b:
                y += 1
            else:
                break
        if y == n_j: continue
        res += (v-u-trend_i) * (y-x-trend_j) - ((b-a)*C - 2*J)
        # check if this step is admissible
        #if y < n_j and x > 0 and v < n_i and u > 0:
        #    count += 1
        #    res += (v-u-trend_i) * (y-x-trend_j)
    #if count < n_k and count > 0:
    #    res *= n_k * 1. / count
    res /= T
    return res

@autojit
def I_ij(Z_i,Z_j,H,T,L_j):
    """
    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals
    \frac{1}{T} \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} [ (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^j ]
    """
    n_i = Z_i.shape[0]
    n_j = Z_j.shape[0]
    res = 0
    u = 0
    #count = 0
    trend_j = .5 * (H**2) * L_j
    for t in range(n_i):
        tau = Z_i[t]
        tau_minus_H = tau - H
        if tau_minus_H < 0: continue
        while u < n_j:
            if Z_j[u] <= tau_minus_H :
                u += 1
            else:
                break
        v = u
        sub_res = 0.
        while v < n_j:
            tau_minus_tau_p = tau - Z_j[v]
            if tau_minus_tau_p > 0:
                sub_res += tau_minus_tau_p
                #count += 1
                v += 1
            else:
                break
        if v == n_j: continue
        res += sub_res - trend_j
    #if count < n_i and count > 0:
    #    res *= n_i * 1. / count
    res /= T
    return res


if __name__ == "__main__":
    import gzip, pickle
    #filename = '/data/users/achab/nphc/datasets/rect/rect_d10_nonsym_2_log10T8_with_params_094.pkl.gz'
    filename = '/Users/massil/Programmation/git/nphc/test.pkl.gz'
    f = gzip.open(filename)
    data = pickle.load(f)
    f.close()
    cumul = Cumulants(data[:3])
    one_day_cumul = Cumulants(data[0])
    cumul.set_C(H=5)
    C_H5 = cumul.C
    cumul.set_C(H=20,method='parallel')
    C_H20 = cumul.C
    print("Same C for different H ?")
    print(np.allclose(C_H5,C_H20))
    cumul.set_J(H=5)
    J_H5 = cumul.J
    cumul.set_J(H=20)
    J_H20 = cumul.J
    print("Same J for different H ?")
    print(np.allclose(J_H5,J_H20))
    cumul.set_E_c(H=5)
    E_c_H5 = cumul.E_c
    cumul.set_E_c(H=20)
    E_c_H20 = cumul.E_c
    print("Same E_c for different H ?")
    print(np.allclose(E_c_H5,E_c_H20))

from numba import autojit, jit, double, int32, int64, float64
from scipy.linalg import inv, pinv, eigh
from joblib import Parallel, delayed
from math import sqrt, pi, exp, cos
from scipy.stats import norm
import numpy as np


class Cumulants(object):

    def __init__(self, realizations=[], half_width=100.):
        if all(isinstance(x, list) for x in realizations):
            self.realizations = realizations
        else:
            self.realizations = [realizations]
        self.dim = len(self.realizations[0])
        self.n_realizations = len(self.realizations)
        self.time = np.zeros(self.n_realizations)
        for day, realization in enumerate(self.realizations):
            T_day = float(max(x[-1] for x in realization)) - float(min(x[0] for x in realization))
            self.time[day] = T_day
        self.L = np.zeros((self.n_realizations, self.dim))
        self.C = np.zeros((self.n_realizations, self.dim, self.dim))
        self._J = np.zeros((self.n_realizations, self.dim, self.dim))
        self._E_c = np.zeros((self.n_realizations, self.dim, self.dim, 2))
        self.K_c = np.zeros((self.n_realizations, self.dim, self.dim))
        self.L_th = None
        self.C_th = None
        self.K_c_th = None
        self.R_true = None
        self.mu_true = None
        self.half_width = half_width

    # ###########
    # ## Decorator to compute the cumulants on each day, and average
    # ###########
    #
    # def average_if_list_of_multivariate_processes(func):
    #     def average_cumulants(self, *args, **kwargs):
    #         if getattr(self, 'N_is_list_of_multivariate_processes', False):
    #             # if self.realizations_is_list_of_multivariate_processes:
    #             for n, multivar_process in enumerate(self.realizations):
    #                 cumul = Cumulants(N=multivar_process)
    #                 res_one_process = func(cumul, *args, **kwargs)
    #                 if n == 0:
    #                     res = np.zeros_like(res_one_process)
    #                 res += res_one_process
    #             res /= n + 1
    #         else:
    #             res = func(self, *args, **kwargs)
    #         return res
    #
    #     return average_cumulants

    #########
    ## Functions to compute third order cumulant
    #########

    def compute_L(self):
        for day, realization in enumerate(self.realizations):
            L = np.zeros(self.dim)
            for i in range(self.dim):
                process = realization[i]
                if process is None:
                    L[i] = -1.
                else:
                    L[i] = len(process) / self.time[day]
            self.L[day] = L.copy()

    # def compute_C(self, half_width=0., method='parallel', filter='rectangular', sigma=1.0):
    #     if half_width == 0.:
    #         h_w = self.half_width
    #     else:
    #         h_w = half_width
    #     d = self.dim
    #
    #     for day in range(len(self.realizations)):
    #         realization = self.realizations[day]
    #         if method == 'classic':
    #             C = np.zeros((d, d))
    #             for i in range(d):
    #                 for j in range(d):
    #                     C[i, j] = A_ij_rect(realization[i], realization[j], -h_w, h_w, self.time[day], self.L[day][j],
    #                                    filter=filter, sigma=sigma)
    #         elif method == 'parallel':
    #             l = Parallel(-1)(
    #                     delayed(A_ij_rect)(realization[i], realization[j], -h_w, h_w, self.time[day], self.L[day][j],
    #                                   filter=filter, sigma=sigma)
    #                     for i in range(d) for j in range(d))
    #             C = np.array(l).reshape(d, d)
    #         # we keep the symmetric part to remove edge effects
    #         C[:] = 0.5 * (C + C.T)
    #         self.C[day] = C.copy()


    def compute_C_and_J(self, half_width=0., method='parallel', filter='rectangular', sigma=1.0):
        if half_width == 0.:
            h_w = self.half_width
        else:
            h_w = half_width
        d = self.dim

        if filter == "rectangular":

            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                if method == 'classic':
                    C = np.zeros((d,d))
                    J = np.zeros((d, d))
                    for i in range(d):
                        for j in range(d):
                            z = A_and_I_ij_rect(realization[i], realization[j], h_w, self.time[day], self.L[day][j])
                            C[i,j] = z.real
                            J[i,j] = z.imag
                elif method == 'parallel':
                    l = Parallel(-1)(
                            delayed(A_and_I_ij_rect)(realization[i], realization[j], h_w, self.time[day], self.L[day][j])
                            for i in range(d) for j in range(d))
                    C_and_J = np.array(l).reshape(d, d)
                    C = C_and_J.real
                    J = C_and_J.imag
                # we keep the symmetric part to remove edge effects
                C[:] = 0.5 * (C + C.T)
                J[:] = 0.5 * (J + J.T)
                self.C[day] = C.copy()
                self._J[day] = J.copy()

        elif filter == "gaussian":

            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                if method == 'classic':
                    J = np.zeros((d, d))
                    for i in range(d):
                        for j in range(d):
                            J[i, j] = I_ij(realization[i], realization[j], h_w, self.time[day], self.L[day][j],
                                           filter=filter,
                                           sigma=sigma)
                elif method == 'parallel':
                    l = Parallel(-1)(
                            delayed(I_ij)(realization[i], realization[j], h_w, self.time[day], self.L[day][j],
                                          filter=filter,
                                          sigma=sigma)
                            for i in range(d) for j in range(d))
                    J = np.array(l).reshape(d, d)
                # we keep the symmetric part to remove edge effects
                J[:] = 0.5 * (J + J.T)
                self._J[day] = J.copy()

        else:

            raise ValueError(
                "In `compute_C_and_J`: the filtering function should be either `rectangular` or `gaussian`.")

    def compute_E_c(self, half_width=0., method='parallel', filter='rectangular', sigma=1.0):
        if half_width == 0.:
            h_w = self.half_width
        else:
            h_w = half_width
        d = self.dim

        if filter == "rectangular":

            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                E_c = np.zeros((d, d, 2))
                if method == 'classic':
                    for i in range(d):
                        for j in range(d):
                            E_c[i, j, 0] = E_ijk_rect(realization[i], realization[j], realization[j], -h_w, h_w,
                                                      self.time[day], self.L[day][i], self.L[day][j], self.C[day][i, j],
                                                      self._J[day][i, j])
                            E_c[i, j, 1] = E_ijk_rect(realization[j], realization[j], realization[i], -h_w, h_w,
                                                      self.time[day], self.L[day][j], self.L[day][j], self.C[day][j, j],
                                                      self._J[day][j, j])
                elif method == 'parallel':
                    l1 = Parallel(-1)(
                            delayed(E_ijk_rect)(realization[i], realization[j], realization[j], -h_w, h_w,
                                                self.time[day], self.L[day][i], self.L[day][j], self.C[day][i, j],
                                                self._J[day][i, j]) for i in range(d) for j in range(d))
                    l2 = Parallel(-1)(
                            delayed(E_ijk_rect)(realization[j], realization[j], realization[i], -h_w, h_w,
                                                self.time[day], self.L[day][j], self.L[day][j], self.C[day][j, j],
                                                self._J[day][j, j]) for i in range(d) for j in range(d))
                    E_c[:, :, 0] = np.array(l1).reshape(d, d)
                    E_c[:, :, 1] = np.array(l2).reshape(d, d)
                self._E_c[day] = E_c.copy()

        elif filter == "gaussian":

            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                E_c = np.zeros((d, d, 2))
                if method == 'classic':
                    for i in range(d):
                        for j in range(d):
                            E_c[i, j, 0] = E_ijk_gauss(realization[i], realization[j], realization[j], -h_w, h_w,
                                                       self.time[day], self.L[day][i], self.L[day][j], sigma=sigma)
                            E_c[i, j, 1] = E_ijk_gauss(realization[j], realization[j], realization[i], -h_w, h_w,
                                                       self.time[day], self.L[day][j], self.L[day][j], sigma=sigma)
                elif method == 'parallel':
                    l1 = Parallel(-1)(
                            delayed(E_ijk_gauss)(realization[i], realization[j], realization[j], -h_w, h_w,
                                                 self.time[day], self.L[day][i], self.L[day][j], sigma=sigma) for i in
                            range(d) for j in range(d))
                    l2 = Parallel(-1)(
                            delayed(E_ijk_gauss)(realization[j], realization[j], realization[i], -h_w, h_w,
                                                 self.time[day], self.L[day][j], self.L[day][j], sigma=sigma) for i in
                            range(d) for j in range(d))
                    E_c[:, :, 0] = np.array(l1).reshape(d, d)
                    E_c[:, :, 1] = np.array(l2).reshape(d, d)
                self._E_c[day] = E_c.copy()

        else:

            raise ValueError("In `compute_E_c`: the filtering function should be either `rectangular` or `gaussian`.")

    def set_R_true(self, R_true):
        self.R_true = R_true

    def set_mu_true(self, mu_true):
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
        self.K_c_th = get_K_c_th(self.L_th, self.C_th, self.R_true)

    def compute_cumulants(self, half_width=0., method="parallel", filter='rectangular', sigma=1.0):
        self.compute_L()
        print("L is computed")
        # self.compute_C(half_width=half_width, method=method, filter=filter, sigma=sigma)
        self.compute_C_and_J(half_width=half_width, method=method, filter=filter, sigma=sigma)
        print("C is computed")
        self.compute_E_c(half_width=half_width, method=method, filter=filter, sigma=sigma)
        self.K_c = [get_K_c(self._E_c[day]) for day in range(self.n_realizations)]
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
    K_c = np.zeros_like(E_c[:, :, 0])
    K_c += 2 * E_c[:, :, 0]
    K_c += E_c[:, :, 1]
    K_c /= 3.
    return K_c


##########
## Theoretical cumulants L, C, K, K_c
##########

@autojit
def get_L_th(mu, R):
    return np.dot(R, mu)


@autojit
def get_C_th(L, R):
    return np.dot(R, np.dot(np.diag(L), R.T))


@autojit
def get_K_c_th(L, C, R):
    d = len(L)
    if R.shape[0] == d ** 2:
        R_ = R.reshape(d, d)
    else:
        R_ = R.copy()
    K_c = np.dot(C, (R_ * R_).T)
    K_c += 2 * np.dot(R_, (R_ * C).T)
    K_c -= 2 * np.dot(np.dot(R_, np.diag(L)), (R_ * R_).T)
    return K_c


##########
## Useful fonctions to set_ empirical integrated cumulants
##########

@autojit
def filter_fun(X, sigma, filter='rectangular'):
    if filter == 'rectangular':
        return np.ones_like(X)
    elif filter == 'gaussian':
        return sigma * sqrt(2 * pi) * norm.pdf(X, scale=sigma)
    else:
        return np.zeros_like(X)


# @jit(double(double[:],double[:],int32,int32,double,double,double), nogil=True, nopython=True)
# @jit(float64(float64[:],float64[:],int64,int64,int64,float64,float64), nogil=True, nopython=True)
@autojit
def A_ij_rect(realization_i, realization_j, a, b, T, L_j):
    """
    Computes the mean centered number of jumps of N^j between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^i} ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j (b - a) )
    """
    res = 0
    u = 0
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]

    trend_j = L_j * (b - a)

    for t in range(n_i):
        # count the number of jumps
        tau = realization_i[t]
        if tau + a < 0: continue
        while u < n_j:
            if realization_j[u] <= tau + a:
                u += 1
            else:
                break

        v = u
        while v < n_j:
            if realization_j[v] < tau + b:
                v += 1
            else:
                break
        if v == n_j: continue
        res += v - u - trend_j
    res /= T
    return res


@autojit
def A_ij_gauss(realization_i, realization_j, a, b, T, L_j, sigma=1.0):
    """
    Computes the mean centered number of jumps of N^j between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^i} ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j (b - a) )
    """
    res = 0
    u = 0
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]

    trend_j = L_j * sigma * sqrt(2 * pi) * (norm.sf(a) - norm.sf(b))

    for t in range(n_i):
        # count the number of jumps
        tau = realization_i[t]
        if tau + a < 0: continue
        while u < n_j:
            if realization_j[u] <= tau + a:
                u += 1
            else:
                break
        time_delta = np.zeros(n_j)
        indicator = np.zeros(n_j)
        v = u
        while v < n_j:
            if realization_j[v] < tau + b:
                time_delta[v] = realization_j[v] - tau
                indicator[v] = 1.
                v += 1
            else:
                break
        if v == n_j: continue
        filtered_times = filter_fun(time_delta, sigma, filter=filter)
        delta = np.dot(indicator, filtered_times)
        res += delta - trend_j
    res /= T
    return res


@autojit
def E_ijk_rect(realization_i, realization_j, realization_k, a, b, T, L_i, L_j, C_ij, J_ij):
    """
    Computes the mean of the centered product of i's and j's jumps between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^k} ( N^i_{\tau + b} - N^i_{\tau + a} - \Lambda^i * ( b - a ) )
                                  * ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j * ( b - a ) )
    """
    res = 0
    u = 0
    x = 0
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    n_k = realization_k.shape[0]

    trend_i = L_i * (b - a)
    trend_j = L_j * (b - a)

    for t in range(n_k):
        tau = realization_k[t]

        if tau + a < 0: continue

        # work on realization_i
        while u < n_i:
            if realization_i[u] <= tau + a:
                u += 1
            else:
                break
        v = u

        while v < n_i:
            if realization_i[v] < tau + b:
                v += 1
            else:
                break

        # work on realization_j
        while x < n_j:
            if realization_j[x] <= tau + a:
                x += 1
            else:
                break
        y = x

        while y < n_j:
            if realization_j[y] < tau + b:
                y += 1
            else:
                break
        if y == n_j or v == n_i: continue

        res += (v - u - trend_i) * (y - x - trend_j) - ((b - a) * C_ij - 2 * J_ij)
    res /= T
    return res


@autojit
def E_ijk_gauss(realization_i, realization_j, realization_k, a, b, T, L_i, L_j, C_ij, J_ij, sigma=1.0):
    """
    Computes the mean of the centered product of i's and j's jumps between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^k} ( N^i_{\tau + b} - N^i_{\tau + a} - \Lambda^i * ( b - a ) )
                                  * ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j * ( b - a ) )
    """
    res = 0
    u = 0
    x = 0
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    n_k = realization_k.shape[0]

    trend_i = L_i * (b - a)
    trend_j = L_j * (b - a)

    for t in range(n_k):
        tau = realization_k[t]

        if tau + a < 0: continue
        # work on realization_i
        while u < n_i:
            if realization_i[u] <= tau + a:
                u += 1
            else:
                break
        v = u
        time_delta_i = np.zeros(n_i)
        indicator_i = np.zeros(n_i)
        while v < n_i:
            if realization_i[v] < tau + b:
                time_delta_i[v] = realization_i[v] - tau
                indicator_i[v] = 1.
                v += 1
            else:
                break
        if v == n_i: continue

        # work on realization_j
        while x < n_j:
            if realization_j[x] <= tau + a:
                x += 1
            else:
                break
        y = x
        time_delta_j = np.zeros(n_j)
        indicator_j = np.zeros(n_j)
        while y < n_j:
            if realization_j[y] < tau + b:
                time_delta_j[y] = realization_j[y] - tau
                indicator_j[y] = 1.
                y += 1
            else:
                break
        if y == n_j: continue
        filtered_times_i = filter_fun(time_delta_i, sigma, filter=filter)
        delta_i = np.dot(indicator_i, filtered_times_i)
        filtered_times_j = filter_fun(time_delta_j, sigma, filter=filter)
        delta_j = np.dot(indicator_j, filtered_times_j)
        res += (delta_i - trend_i) * (delta_j - trend_j) - ((b - a) * C_ij - 2 * J_ij)
    res /= T
    return res


@autojit
def I_ij(realization_i, realization_j, half_width, T, L_j, filter='rectangular', sigma=1.0):
    """
    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals
    \frac{1}{T} \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} [ (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^j ]
    """
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    res = 0
    u = 0
    if filter == 'rectangular':
        trend_j = .5 * (half_width ** 2) * L_j
    elif filter == 'gaussian':
        trend_j = sigma ** 2 * (1 - exp(-.5 * (half_width / sigma) ** 2)) * L_j

    for t in range(n_i):
        tau = realization_i[t]
        tau_minus_half_width = tau - half_width
        if tau_minus_half_width < 0: continue
        while u < n_j:
            if realization_j[u] <= tau_minus_half_width:
                u += 1
            else:
                break
        v = u
        sub_res = 0.
        while v < n_j:
            tau_minus_tau_p = tau - realization_j[v]
            if tau_minus_tau_p > 0:
                sub_res += tau_minus_tau_p
                v += 1
            else:
                break
        if v == n_j: continue
        res += sub_res - trend_j
    res /= T
    return res


@autojit
def A_and_I_ij_rect(realization_i, realization_j, half_width, T, L_j):
    """
    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals
    \frac{1}{T} \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} [ (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^j ]
    """
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    res_C = 0
    res_J = 0
    u = 0
    trend_C_j = L_j * 2 * half_width
    trend_J_j = .5 * (half_width ** 2) * L_j

    for t in range(n_i):
        tau = realization_i[t]
        tau_minus_half_width = tau - half_width
        if tau_minus_half_width < 0: continue
        while u < n_j:
            if realization_j[u] <= tau_minus_half_width:
                u += 1
            else:
                break
        v = u
        w = u
        sub_res = 0.

        tau_plus_half_width = tau + half_width
        while w < n_j:
            if realization_j[w] < tau_plus_half_width:
                w += 1
            else:
                break

        while v < n_j:
            tau_minus_tau_p = tau - realization_j[v]
            if tau_minus_tau_p > 0:
                sub_res += tau_minus_tau_p
                v += 1
            else:
                break
        if v == n_j or w == n_j: continue
        res_C += w - u - trend_C_j
        res_J += sub_res - trend_J_j
    res_C /= T
    res_J /= T
    return res_C + res_J * 1j


@autojit
def A_and_I_ij_gauss(realization_i, realization_j, half_width, T, L_j, sigma=1.0):
    """
    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals
    \frac{1}{T} \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} [ (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^j ]
    """
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    res_C = 0
    res_J = 0
    u = 0
    
    trend_C_j = L_j * sigma * sqrt(2 * pi) * (norm.sf(-half_width) - norm.sf(half_width))
    trend_J_j = sigma ** 2 * (1 - exp(-.5 * (half_width / sigma) ** 2)) * L_j

    for t in range(n_i):
        tau = realization_i[t]
        tau_minus_half_width = tau - half_width
        if tau_minus_half_width < 0: continue
        while u < n_j:
            if realization_j[u] <= tau_minus_half_width:
                u += 1
            else:
                break
        v = u
        w = u
        sub_res = 0.

        while v < n_j:
            tau_minus_tau_p = tau - realization_j[v]
            if tau_minus_tau_p > 0:
                sub_res += tau_minus_tau_p
                v += 1
            else:
                break

        tau_plus_half_width = tau + half_width
        while w < n_j:
            if realization_j[v] < tau_plus_half_width:
                w += 1
            else:
                break

        if v == n_j or w == n_j: continue
        res_C += (w - u) - trend_C_j
        res_J += sub_res - trend_J_j
    res_C /= T
    res_J /= T
    return res_C, res_J

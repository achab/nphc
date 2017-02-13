# We here describe how to compute the empirical covariance density of a
# multivariate point process, using a function implemented in nphc/utils/cumulants.py.

import numpy as np
from nphc.cumulants import A_ij_rect
from joblib import Parallel, delayed

def cov_density(realization_i, realization_j, T, L_j, log_start=0., log_end=3, n_points=100):
    H_range = np.logspace(log_start, log_end, n_points)
    Z = Parallel(-1)(delayed(A_ij_rect)(realization_i, realization_j, -H, H, T, L_j) for H in H_range)
    Z = np.array(Z)
    X = H_range[:-1]
    Y = np.diff(Z) / np.diff(H_range)
    return X, Y

if __name__ == "__main__":
    import mlpp.simulation as hk
    import matplotlib.pyplot as plt

    T = 1e7

    hawkes = hk.SimuHawkes(n_nodes=1, end_time=T, verbose=False)
    kernel = hk.HawkesKernelExp(1 / 4, 4)
    hawkes.set_kernel(0, 0, kernel)
    hawkes.set_baseline(0, 0.01)
    hawkes.simulate()

    N = hawkes.timestamps[0]
    L = len(N) / T
    X, Y = cov_density(N, N, T, L)

    plt.plot(X,Y)
    plt.xscale("log")
    plt.savefig("covariance_exp.png")
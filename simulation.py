import numpy as np
import scipy as sp
import mlpp.pp.hawkes as hk
from numba import autojit

@autojit
def simulate_mu(d, mu=None):
    if mu is None:
        return np.random.rand(d)
    else:
        return mu * np.ones(d)

@autojit
def simulate_A(d, blocks=None, kernel_type='exp'):
    if blocks is not None:
        assert sum(blocks) == d, "The sum of block lengths is not equal to the dimension."
    else:
        blocks = np.ones(d)
    if kernel_type == 'exp':
        L = []
        for x in blocks:
            L.append(np.random.rand(x, x))
        alpha = sp.linalg.block_diag(*L)
    else:
        raise ValueError("This kernel is not implemented yet. Use 'exp' instead.")
    #beta = np.random.rand(d)
    #return [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)] for (a_list, b_list) in zip(alpha, beta)]
    return alpha

@autojit
def simulate(kernels, mus, n=1000):
    h = hk.Hawkes(kernels=kernels, mus=mus)
    h.simulate(n)
    return h

import numpy as np
import scipy
import matplotlib.pyplot as plt
from pylab import rcParams
from mlpp.hawkesnoparam.estim import Estim
import mlpp.pp.hawkes as hk
import admm_hawkes.simulation as simu
from mlpp.base.utils import TimeFunction

# simulation of Hawkes processes
d = 10
mu = 0.005 * np.ones(d)
mus = simu.simulate_mu(d, mu=mu)
blocks = [2,3,5]
L = []
for x in blocks:
    L.append(np.random.rand(x, x))
alpha_truth = scipy.linalg.block_diag(*L) / (d)
print(alpha_truth)
#alpha_truth = simu.simulate_A(d, blocks=[2,3,5])
#alpha_truth = np.arange(1,d**2+1).reshape((d,d)) / (2*d**2)
# check that alpha_truth can generate a stable Hawkes process
_, s, _ = np.linalg.svd(alpha_truth)
print(s.max())
assert s.max() < 1, "alpha_truth cannot generate a stable Hawkes process"
beta = np.arange(1, d**2+1).reshape((d,d)) / (d**2)
kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)] for (a_list, b_list) in zip(alpha_truth, beta)]
h = hk.Hawkes(kernels=kernels, mus=list(mus))
h.simulate(1000000)
estim = Estim(h)

# estimation procedure
from admm_hawkes.solver import admm
import admm_hawkes.prox as prox
#X0 = np.eye(d)
X0 = np.ones(d**2).reshape(d,d)

# main step
X1 = admm(estim, prox.sq_frob, X0, X0, alpha_truth, rho=.1, maxiter=10000)

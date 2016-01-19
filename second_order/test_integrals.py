import numpy as np
import scipy
from mlpp.hawkesnoparam.estim import Estim
import mlpp.pp.hawkes as hk
import whma.simulation as simu
from mlpp.base.utils import TimeFunction
from whma.metrics import rel_err, rank_corr

# simulation of Hawkes processes
d = 10
mu = 0.0005 * np.ones(d)
mus = simu.simulate_mu(d, mu=mu)
blocks = [5,5]
L = []
L.append(np.ones((blocks[0],blocks[0])))
L.append(np.ones((blocks[1],blocks[1])))
Alpha_truth = scipy.linalg.block_diag(*L) / 6
# add noise
#Alpha_truth += 0.01

hMax = 40
hDelta = .01
from math import log
beta_min = log(1000) / hMax
beta_max = log(10./9.) / hDelta

LL = []
LL.append(beta_max*np.ones((blocks[0],blocks[0])))
LL.append(beta_min*np.ones((blocks[1],blocks[1])))
Beta = scipy.linalg.block_diag(*LL)

kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)] for (a_list, b_list) in zip(Alpha_truth, Beta)]
h = hk.Hawkes(kernels=kernels, mus=list(mus))
T_max = 10000
h.simulate(T_max)
estim = Estim(h, n_threads=8, hDelta=hDelta, hMax=hMax)

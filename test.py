import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab import rcParams
from mlpp.hawkesnoparam.estim import Estim
import mlpp.pp.hawkes as hk
from mlpp.base.utils import TimeFunction

# simulation of Hawkes processes
d = 3
mu = 0.005 * np.ones(d)
mus = [m for m in mu]
alpha = np.arange(1,d**2+1).reshape((d,d)) / (2*d**2)
beta = np.arange(1, d**2+1).reshape((d,d)) / (d**2)
kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)] for (a_list, b_list) in zip(alpha, beta)]
h = hk.Hawkes(kernels=kernels, mus=mus)
h.simulate(1000)
estim = Estim(h)

# estimation procedure
from admm_hawkes.solver import admm
import admm_hawkes.prox as prox
X0 = np.eye(d)

# main step
X1 = admm(estim=estim, prox_fun=prox.l1, X1_0=X0, X4_0=X0)

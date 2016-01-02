import numpy as np
from transform import empirical_sqrt_mean, empirical_cross_corr, integrated_claw
from mlpp.hawkesnoparam import Estim


class Cumulants(Estim):

    def __init__(self):
        super().__init__()
        self.L = np.array(self.lam)
        self.C = None
        self.K = None
        self.K_partial = None

    def set_C(self):
        self.C = get_C(self)

    def set_K(self):
        self.K = get_K(self)

    def set_K_partial(self):
        self.K_partial = get_K_partial(self)


def get_C(estim):
    G = integrated_claw(estim, method='lin')
    C = np.einsum('i,ij->ij', estim.L, G.T)
    # the following line cancels the edge effects
    C = .5 * (C + C.T)
    return C

def get_K(estim):
    pass

def get_K_partial(estim):
    pass


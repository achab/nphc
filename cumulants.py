import numpy as np
from transform import empirical_sqrt_mean, empirical_cross_corr, integrated_claw
from mlpp.hawkesnonparam import Estim


class Cumulants(Estim):

    def __init__(self):
        super().__init__()
        self.L = np.array(self.lam)
        diagD, O = empirical_cross_corr(self)
        self.C = np.dot(O,np.dot(np.diag(diagD),O.T))
        self.K = None

    def set_K(self):
        self.K = ord3cumul(self)

    def get_corr(self):
        G = integrated_claw(self, method='lin')
        C = np.einsum('i,ij->ij', self.L, G.T)
        # the following line cancels the edge effects
        C = .5 * (C + C.T)
        return C


def ord3cumul(estim):
    pass


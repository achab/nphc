import numpy as np
from .transform import empirical_sqrt_mean, empirical_cross_corr, integrated_claw
from mlpp.hawkesnoparam import Estim


class Cumulants(Estim):

    def __init__(self,*args,**kwargs):
        super(self.__class__, self).__init__(self,*args,**kwargs)
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
    np.fill_diagonal(G, G.diagonal()+1)
    C = np.einsum('i,ij->ij', estim.lam, G.T)
    # the following line cancels the edge effects
    C = .5 * (C + C.T)
    return C

def get_K(cumul):
    pass

def get_K_partial(L, C, Integrals):
#    L = np.array(estim.lam)
#    K_part = np.array(estim.integrals)
    K_part = Integrals.copy()
    K_part -= np.einsum('ii,j->ij',C,L)
    K_part -= 2*np.einsum('ij,i->ij',C,L)
    K_part += 2*np.einsum('i,j->ij',L**2,L)
    return K_part


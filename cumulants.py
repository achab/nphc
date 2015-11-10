import numpy as np
from numpy.polynomial.legendre import leggauss
from numba import jit


# Computation of integral of claw
def compute_ints_claw(estim, n_quad=50, xmax=40, method='gauss'):
    """ (Estim) -> float
    Computes the integral of the conditional law ij at the
    difference of quadrature points using linear interpolation

    The following code is borrowed from method `compute` in mlpp/hawkesnoparam/estim.py
    """
    # Find closest value of a signal

    def lin0(sig, t):
        (x,y) = sig
        if (t >= x[-1]): return 0
        index = np.searchsorted(x,t)
        if index == len(y)-1: return y[index]
        if np.abs(x[index]-t) < np.abs(x[index+1]-t):
            return y[index]
        else:
            return y[index+1]

    def linc(sig, t):
        (x,y) = sig
        if (t >= x[-1]): return y[-1]
        index = np.searchsorted(x,t)
        if np.abs(x[index]-t) < np.abs(x[index+1]-t):
            return y[index]
        else:
            return y[index+1]

    if method == 'gauss':
        estim.quad_x, estim.quad_w = leggauss(n_quad)
        estim.quad_x = xmax*(estim.quad_x+1)/2
        estim.quad_w *= xmax/2
    elif method == 'lin':
        x1=np.arange(0.,xmax,xmax/n_quad)
        estim.quad_x = x1
        estim.quad_w = estim.quad_x[1:]-estim.quad_x[:-1]
        estim.quad_w = np.append(estim.quad_w,estim.quad_w[-1])
        #n_quad = len(estim.quad_x)
        #estim.quad_x = np.array(estim.quad_x)
        #estim.quad_w = np.array(estim.quad_w)
    else:
        pass

    # Call the function _compute_ints_claw
    estim._compute_ints_claw()

    # Returns the value of a claw at a point
    def G(i, j, l, t):
        if t < 0:
            print("G(): should not be called for t< 0")
        index = estim._ijl2index[i][j][l]
        return lin0(estim._claw[index],t)

    # Returns the integral of a claw between t1 and t2
    def DIG(i, j, l, t1, t2):
        if t1 >= t2:
            print("t2>t1 faux dans IG")
        index = estim._ijl2index[i][j][l]
        return linc(estim.IG[index],t2)-linc(estim.IG[index],t1)

    # Returns the integral of x times a claw between t1 and t2
    def DIG2(i, j, l, t1, t2):
        if t1 >= t2:
            print("t2>t1 faux dans IG2")
        index = estim._ijl2index[i][j][l]
        return linc(estim.IG2[index],t2)-linc(estim.IG2[index],t1)

    # Fill the matrix to return with integrated claws
    @jit
    def 


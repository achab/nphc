import numpy as np
from scipy.linalg import sqrt, inv
from numpy.polynomial.legendre import leggauss


# Computation of \Sigma^{1/2}
def empirical_mean(estim):
    Sigma_sq_root = sqrt(inv(np.diag(estim)))
    return Sigma_sq_root

# Computation of ||c||^{-1/2}
def empirical_cross_corr(estim):
    pass

# Computation of integral of claw
def compute_ints_claw(estim, n_quad=50, xmax=40, method='gauss'):
    """ (Estim) -> float
    Computes the integral of the conditional law ij at the
    difference of quadrature points using linear interpolation

    The following code is borrowed from method `compute`in mlpp/hawkesnoparam/estim.py
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
        quad_x, quad_w = leggauss(n_quad)
        quad_x = xmax*(quad_x+1)/2
        quad_w *= xmax/2
    elif method == 'lin':
        x1=np.arange(0.,xmax,xmax/n_quad)
        quad_x = x1
        quad_w = quad_x[1:]-quad_x[:-1]
        quad_w = np.append(quad_w,quad_w[-1])
        #n_quad = len(quad_x)
        #quad_x = np.array(quad_x)
        #quad_w = np.array(quad_w)
    else:
        pass



# # the following function is incomplete
# def inspector(loss_fun, x_real, n_iter, norm, verbose=False):
#     """A closure called to update metrics after each iteration."""
#     objectives = []
#     errors = []
#     it = [0]  # This is a hack to be able to modify 'it' inside the closure.
#     def inspector_cl(xk):
#         obj = loss_fun(xk)
#         err = norm(xk - x_real) / norm(x_real)
#         objectives.append(obj)
#         errors.append(err)
#         if verbose == True:
#             if it[0] == 0:
#                 print ' | '.join([name.center(8) for name in ["it", "obj", "err"]])
#             if it[0] % (n_iter / 5) == 0:
#                 print ' | '.join([("%d" % it[0]).rjust(8), ("%.2e" % obj).rjust(8), ("%.2e" % err).rjust(8)])
#             it[0] += 1
#     inspector_cl.obj = objectives
#     inspector_cl.err = errors
#     return inspector_cl
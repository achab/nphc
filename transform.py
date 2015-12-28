import numpy as np
from scipy.linalg import eig, inv
from numpy.polynomial.legendre import leggauss
from numba import double, int_, jit, autojit



# Computation of \Sigma^{1/2}
#@autojit
def empirical_sqrt_mean(lam):
    return np.sqrt(lam)

# Eigen-decomposition of ||C||
#@autojit
def empirical_cross_corr(estim):
    G = integrated_claw(estim, method='lin')
    np.fill_diagonal(G, G.diagonal()+1)
    C = np.einsum('i,ij->ij', np.array(estim.lam), G.T)
    # THE FOLLOWING LINE IS A TOTAL HACK
    C = .5 * (C + C.T)
    # C should be symmetric
    assert np.allclose(C, C.T), "C should be symmetric !"
    diagD, O = eig(C)
    # we cast the imaginary part since it equals zero
    diagD = np.array([x.real for x in diagD])
    # O should be orthogonal
    assert np.allclose(O.T, inv(O)), "O should be an orthogonal matrix !"
    return diagD, O

# Computation of \hat{\nu}(0)
#@autojit
def corr_matrix(estim):
    G = integrated_claw(estim, method='lin')
    C = np.einsum('i,ij->ij', np.array(estim.lam), G.T)
    # THE FOLLOWING LINE IS A TOTAL HACK
    C = .5 * (C + C.T)
    # C should be symmetric
    return C


# Computation of integral of claw
def integrated_claw(estim, n_quad=50, xmax=40, method='gauss'):
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
    def fill_matrix_integrated_claw(d):
        C = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                C[i][j] = DIG(i, j, 0, 0, xmax)
        return C

    fast_fill_matrix_integrated_claw = jit(double[:,:](int_))(fill_matrix_integrated_claw)

    return fast_fill_matrix_integrated_claw(estim._dim)



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
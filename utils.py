import numpy as np
from scipy.linalg import sqrtm
from numba import autojit
from cumulants import integrated_claw


# Computation of \Sigma^{1/2}
@autojit
def empirical_sqrt_mean(lam):
    return np.diag(np.sqrt(lam))

# Computation of ||C||^{1/2}
@autojit
def empirical_sqrt_cross_corr(estim):
    G = integrated_claw(estim)
    np.fill_diagonal(G, G.diagonal()+1)
    C = np.einsum('i,ij->ij', np.array(estim.lam), G.T)
    return sqrtm(C)


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
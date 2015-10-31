from scipy.linalg import sqtm, inv

# Computation of \Sigma^{1/2}
def empirical_mean(estim):
    Sigma_sq_root = sqrt(inv(np.diag(estim)))
    return Sigma_sq_root

# Computation of ||c||^{-1/2}
def empirical_cross_corr(estim):
    pass

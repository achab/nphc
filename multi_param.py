from mlpp.optim.model import ModelHawkesFixedExpKernLogLik
from nphc.utils.cumulants import Cumulants
from scipy.optimize import minimize
from multiprocessing import Pool
import gzip, pickle
import numpy as np
import os.path

def worker(kernel_mode_log10T,learning_rate=10.,training_epochs=1000,display_step=200):

    kernel, mode, log10T = kernel_mode_log10T
    i = kernel.find('_d10')
    filename = 'datasets/' + kernel[:i] + '/{}_{}_log10T{}_with_params_000.pkl.gz'.format(kernel, mode, log10T)
    if not os.path.isfile(filename):
        filename = 'datasets/' + kernel[:i] + '/{}_{}_log10T{}_with_params_000.pkl.gz'.format(kernel, mode, log10T+1)

    try:
        f = gzip.open(filename, 'rb')
        data = pickle.load(f)
        f.close()

        cumul, Alpha, Beta, Gamma = data

        beta = Beta[-1,-1]

        if beta > 0:

            ticks = cumul.N
            d = cumul.dim

            model = ModelHawkesFixedExpKernLogLik(beta).fit(ticks)
            bnds = [(.01, None) for _ in range(model.n_coeffs)]
            result = minimize(lambda x: model.loss(x), 1*np.ones(model.n_coeffs), method='L-BFGS-B', bounds=bnds)
            res = result['x'][d:].reshape(d,d)

            import gzip, pickle
            f = gzip.open('results_beta1_{}_{}.pkl.gz'.format(kernel,mode), 'wb')
            pickle.dump(res,f,protocol=2)
            f.close()

    except ImportError:
        print("ImportError for ",filename)



if __name__ == '__main__':

    kernels = ['exp_d10', 'rect_d10', 'plaw_d10']
    modes = ['nonsym_1', 'nonsym_1_hard', 'nonsym_2', 'nonsym_2_hard']
    log10T = [8]
    from itertools import product
    L = list(product(kernels,modes,log10T))
    L.remove(('exp_d10','nonsym_2',log10T[0]))
    L.remove(('exp_d10','nonsym_2_hard',log10T[0]))
    L.remove(('plaw_d10','nonsym_2',log10T[0]))
    L.remove(('plaw_d10','nonsym_2_hard',log10T[0]))

    pool = Pool(10)
    results = pool.map(worker, L)

    import gzip, pickle
    f = gzip.open('results.pkl.gz', 'wb')
    pickle.dump((L,results),f,protocol=2)
    f.close()

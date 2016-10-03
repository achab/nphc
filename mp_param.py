from mlpp.optim.model import ModelHawkesFixedExpKernLogLik
from scipy.optimize import minimize
from mlpp.optim.prox import ProxZero
from mlpp.optim.solver import BFGS
from multiprocessing import Pool
from math import log10
import numpy as np
import os.path

def worker(kernel_mode_log10T_factor):

    kernel, mode, log10T, factor = kernel_mode_log10T_factor
    i = kernel.find('_d20')
    filename = 'datasets/' + kernel[:i] + '/{}_{}_log10T{}_with_params_000.pkl.gz'.format(kernel, mode, log10T)
    if not os.path.isfile(filename):
        filename = 'datasets/' + kernel[:i] + '/{}_{}_log10T{}_with_params_000.pkl.gz'.format(kernel, mode, log10T+1)

    try:
        import gzip, pickle
        f = gzip.open(filename, 'rb')
        data = pickle.load(f)
        f.close()

        cumul, Alpha, Beta, Gamma = data

        if 'nonsym_2' in mode:
            eta = factor*Gamma[15,10]
        else:
            eta = factor*Beta[15,10]

        if eta > 0:

            ticks = cumul.N
            d = cumul.dim

            model = ModelHawkesFixedExpKernLogLik(eta).fit(ticks)
            bnds = [(0.00001, None) for _ in range(model.n_coeffs)]
            result = minimize(lambda x: model.loss(x), np.ones(model.n_coeffs), bounds=bnds) 
            res = result['x'][d:].reshape(d,d)
            
            #solver = BFGS(tol=0, max_iter=max_iter, verbose=True, print_every=10, record_every=1)
            #prox = ProxZero()
            #solver.set_model(model)
            #solver.set_prox(prox)
            #coeffs = solver.solve(np.ones(model.n_coeffs))
            #res = coeffs.reshape(d,d)
            
            if 'nonsym_2' in mode:
                 file_to_write = 'results_gamma{}_{}_{}.pkl.gz'.format(int(log10(factor)),kernel,mode)
            else:
                file_to_write = 'results_beta{}_{}_{}.pkl.gz'.format(int(log10(factor)),kernel,mode)

            import gzip, pickle
            f = gzip.open(file_to_write, 'wb')
            pickle.dump(res,f,protocol=2)
            f.close()

    except ImportError:
        print("ImportError for ",filename)



if __name__ == '__main__':

    kernels = ['exp_d20', 'rect_d20']
    modes = ['nonsym_1_hard', 'nonsym_2_hard']
    log10T = [8]
    factors = [0.001,0.01,0.1,1,10,100,1000]
    from itertools import product
    L = list(product(kernels,modes,log10T,factors))
    #L.remove(('exp_d10','nonsym_2',log10T[0]))
    #L.remove(('exp_d10','nonsym_2_hard',log10T[0]))
    #L.remove(('plaw_d10','nonsym_2',log10T[0]))
    #L.remove(('plaw_d10','nonsym_2_hard',log10T[0]))
    
    pool = Pool(len(L))
    pool.map(worker, L)

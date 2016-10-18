from mlpp.optim.model import ModelHawkesFixedExpKernLogLik
from scipy.optimize import minimize
from mlpp.optim.prox import ProxZero, ProxL1
from mlpp.optim.solver import BFGS, Fista, Ista
from multiprocessing import Pool
from math import log10
import numpy as np
import os.path

#prox = ProxZero()
prox = ProxL1(strength=0)

def worker(kernel_mode_log10T_factor):

    kernel, mode, log10T, factor = kernel_mode_log10T_factor
    i = kernel.find('_d100')
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
            eta = factor*Gamma[60,60]
        else:
            eta = factor*Beta[60,60]

        if eta > 0:

            ticks = cumul.N
            d = cumul.dim

            model = ModelHawkesFixedExpKernLogLik(eta).fit(ticks)
            #bnds = [(0.00001, None) for _ in range(model.n_coeffs)]
            #options = {'maxiter': 5}
            #result = minimize(lambda x: model.loss(x), np.ones(model.n_coeffs), bounds=bnds) 
            #result = minimize(lambda x: model.loss(x), np.ones(model.n_coeffs), method='CG') 
            #res = result['x'][d:].reshape(d,d)
            
            solver = Fista(tol=1e-13, linesearch=True, max_iter=10).set_model(model).set_prox(prox)
            from nphc.main import starting_point
            R0 = starting_point(cumul) 
            from scipy.linalg import inv
            #G0 = np.eye(d) - inv(R0)
            G0 = 0.5*Alpha.copy()
            coeffs0 = np.concatenate([np.ones(d), G0.reshape(d**2,)])
            coeffs = solver.solve(coeffs0)
            res = coeffs.reshape(d,d)
            
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

    kernels = ['exp_d100']
    modes = ['nonsym_1']
    log10T = [8]
    factors = [0.1,10]
    from itertools import product
    L = list(product(kernels,modes,log10T,factors))
    #L.remove(('exp_d10','nonsym_2',log10T[0]))
    #L.remove(('exp_d10','nonsym_2_hard',log10T[0]))
    #L.remove(('plaw_d10','nonsym_2',log10T[0]))
    #L.remove(('plaw_d10','nonsym_2_hard',log10T[0]))
    
    pool = Pool(len(L))
    pool.map(worker, L)

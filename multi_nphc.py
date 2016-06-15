from multiprocessing import Pool
from scipy.linalg import sqrtm
import tensorflow as tf
from nphc.main import NPHC
import numpy as np

def worker(kernel_mode_log10T,learning_rate=10.,training_epochs=1000,display_step=200):

    kernel, mode, log10T = kernel_mode_log10T
    url = 'https://s3-eu-west-1.amazonaws.com/nphc-data/{}_{}_log10T{}_with_params_without_N.pkl.gz'.format(kernel, mode, log10T)
    from utils.loader import load_data
    cumulants = load_data(url)[0]

    d = cumulants.dim

    # Starting point
    sqrt_C = sqrtm(cumulants.C)
    sqrt_L = np.sqrt(cumulants.L)
    initial = tf.constant(np.dot(sqrt_C,np.diag(1./sqrt_L)).astype(np.float32),shape=[d,d])

    R = NPHC(cumulants,initial,alpha=0.5,training_epochs=training_epochs,stochastic=False,weightGMM='eye',\
         display_step=display_step,learning_rate=learning_rate,optimizer='adam')

    return R

if __name__ == '__main__':

    kernels = ['exp_d10', 'rect_d10', 'plaw_d10']
    modes = ['nonsym_1', 'nonsym_1_hard', 'nonsym_2', 'nonsym_2_hard']
    log10T = [10]
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

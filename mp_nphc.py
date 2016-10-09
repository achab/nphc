from multiprocessing import Pool
from scipy.linalg import sqrtm, inv
import tensorflow as tf
from nphc.main import NPHC
import numpy as np

def worker(kernel_mode_log10T,learning_rate=1e1,training_epochs=1001,display_step=200):

    kernel, mode, log10T = kernel_mode_log10T
    url = 'https://s3-eu-west-1.amazonaws.com/nphc-data/{}_{}_log10T{}_with_params_without_N.pkl.gz'.format(kernel, mode, log10T)
    from utils.loader import load_data
    cumulants = load_data(url)[0]

    d = cumulants.dim

    # Starting point
    sqrt_C = sqrtm(cumulants.C)
    sqrt_L = np.sqrt(cumulants.L)
    initial = np.dot(sqrt_C,np.diag(1./sqrt_L))
    
    R = NPHC(cumulants,initial,alpha=0.01,training_epochs=training_epochs,\
         display_step=display_step,learning_rate=learning_rate,optimizer='adam') #,l_l1=0.,l_l2=0.)
    
    G = np.eye(d) - inv(R)
    
    file_to_write = 'results/results_nphc_{}_{}.pkl.gz'.format(kernel,mode)
    
    import gzip, pickle
    f = gzip.open(file_to_write, 'wb')
    pickle.dump(G,f,protocol=2)
    f.close()

if __name__ == '__main__':

    #kernels = ['exp_d10', 'rect_d10', 'plaw_d10']
    kernels = ['plaw_d10']
    #modes = ['nonsym_1', 'nonsym_1_hard', 'nonsym_2', 'nonsym_2_hard']
    modes = ['nonsym_2']
    log10T = [10]
    from itertools import product
    L = list(product(kernels,modes,log10T))

    pool = Pool(10)
    pool.map(worker, L)

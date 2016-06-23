from multiprocessing import Pool
from nphc.utils.simulate_data import args2params, params2kernels, simulate_and_compute_cumul, save
import numpy as np

def ix2str(ix):
    if ix < 10:
        ix_str = '00' + str(ix)
    elif ix < 100:
        ix_str = '0' + str(ix)
    else:
        ix_str = str(ix)
    return ix_str

symmetric = 1
kernel = 'rect'
d = 100
T = 1e6

if symmetric == 0:
    mode = 'd' + str(d) + '_nonsym_1'
elif symmetric == 1:
    mode = 'd' + str(d) + '_nonsym_2'
elif symmetric == 2:
    mode = 'd' + str(d) + '_nonsym_1_hard'
elif symmetric == 3:
    mode = 'd' + str(d) + '_nonsym_2_hard'


def worker(ix):
    ix_str = '_' + ix2str(ix)
    mu, Alpha, Beta, Gamma = args2params(mode, symmetric)
    kernels = params2kernels(kernel, Alpha, Beta, Gamma)
    cumul = simulate_and_compute_cumul(mu, kernels, Alpha, T, 20)
    save(cumul, Alpha, Beta, Gamma, kernel, mode, T, suffix=ix_str)


if __name__ == '__main__':

    indices = np.arange(4)
    pool = Pool(20)
    pool.map(worker,indices)

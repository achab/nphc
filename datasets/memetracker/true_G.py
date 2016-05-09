from multiprocessing import Pool
import pandas as pd
import numpy as np

top500 = pd.read_csv('top500.csv')

url2ix = { i:x for i, x in enumerate(top500['url']) }
ix2url = { x:i for i, x in enumerate(top500['url']) }


def worker(filename):
    """
    Compute the empirical value of || \Phi || by month using the formula
    \int_0^infty \phi^{ij} = N_T^{i \leftarrow j} / N_T^i
    """
    pass


if __name__ == '__main__':

    indices = np.arange(500,dtype=int)

    from glob import glob
    names = glob('df_200*')

    pool = Pool(processes=len(names))
    pool.map(worker,names)

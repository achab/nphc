from multiprocessing import Pool
import numpy as np
import pickle, gzip
from cumulants import Cumulants

def worker(infos):
    """
    Recompute K or K_part for a given dataset
    infos = (filename, mode) with mode \in { 'full', 'part' }
    """
    filename, mode = infos
    f = gzip.open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    if 'plaw' in filename:
        H = 500
    else:
        H = 16
    if mode == 'full':
        data[0].set_all(H)
    else:
        data[0].set_all_part(H)
    ff = gzip.open(filename, 'wb')
    pickle.dump(data,ff,protocol=2)
    ff.close()
    print("Ok for ",filename," with mode ",mode)


if __name__ == '__main__':
    from glob import glob
    filenames = glob('../datasets/*log10T9_with_Beta.pkl.gz')
    modes = ['full', 'part']
    list_infos = []
    for x in filenames:
        for y in modes:
            list_infos.append((x,y))

    pool = Pool(processes=18)
    pool.map(worker,list_infos)
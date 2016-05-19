from multiprocessing import Pool
import numpy as np
import pickle, gzip
from utils.metrics import rel_err
from utils.cumulants import Cumulants


def worker(infos):
    filename, H = infos
    f = gzip.open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    cumul = data[0]
    cumul.set_all_part(H)
    # append errors
    err_C = rel_err(cumul.C_th,cumul.C)
    err_K_part = rel_err(cumul.K_part_th,cumul.K_part)
    print("Done for ",filename," and for = ",H,)
    return filename, H, err_C, err_K_part


if __name__ == '__main__':
    from glob import glob
    h_values = np.logspace(0,4,10)
    filenames = glob('datasets/*log10T9_with_Beta.pkl.gz')
    list_infos = []
    for x in filenames:
        for y in h_values:
            list_infos.append((x,y))

    pool = Pool(processes=18)
    results = pool.map(worker,list_infos)

    import pandas as pd
    pd.DataFrame(results,columns=['name','H','err_C','err_K_part']).to_csv('err_results.csv')

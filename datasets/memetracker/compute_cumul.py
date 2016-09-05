import numpy as np
import pandas as pd
import glob

list_dir_name = ['top10_1months_start2009-04']
#list_dir_name = glob.glob('top10_1months*')

for dir_name in list_dir_name:

    print("Starting for dataset in ",dir_name)

    L = glob.glob(dir_name+'/process_*')
    for x in L:
        if 'with_cumul' in x:
            L.remove(x)


    import gzip, pickle
    N = []
    for x in L:
        f = gzip.open(x,'r')
        process = pickle.load(f,encoding='latin1')
        f.close()
        N.append(process)

    from utils.cumulants import Cumulants
    cumul = Cumulants(N)

    # we set H = 1 hour (in seconds)
    H = 3600
    cumul.hMax = H
    cumul.set_all_part(H)

    ff = gzip.open(dir_name+'/process_with_cumul.pkl.gz','wb')
    pickle.dump(cumul,ff,protocol=2)
    ff.close()

import numpy as np
import pandas as pd
import glob

#list_dir_name = ['top50_1months_start_2008-12']
list_dir_name = glob.glob('top50_1months_start_2009*')

for dir_name in list_dir_name:

    print("Starting for dataset in ",dir_name)

    L = glob.glob(dir_name+'/copy_*')
    for x in L:
        if 'with_cumul' in x:
            L.remove(x)
    L.sort()
    print(len(L))
    import gzip, pickle
    N = []
    for x in L:
        f = gzip.open(x,'r')
        process = pickle.load(f,encoding='latin1')
        f.close()
        N.append(process)

    from nphc.utils.cumulants import Cumulants
    cumul = Cumulants(N)

    # we set H = 1 hour (in seconds)
    H = 3600
    cumul.hMax = H
    cumul.set_all(H)

    ff = gzip.open(dir_name+'/process_with_cumul.pkl.gz','wb')
    pickle.dump(cumul,ff,protocol=2)
    ff.close()

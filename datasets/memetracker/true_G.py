from multiprocessing import Pool
import pandas as pd
import numpy as np
from glob import glob
import gzip, pickle


top500 = pd.read_csv('top500.csv')

url2ix = { i:x for i, x in enumerate(top500['url']) }
ix2url = { x:i for i, x in enumerate(top500['url']) }

names = glob('df_200*')
names.sort()

d = 500

def worker(ind):
    """
    Compute the empirical value of || \Phi || by month using the formula
    \int_0^infty \phi^{ij} = N_T^{i \leftarrow j} / N_T^i
    """
    url = ix2url[ind]
    col_ind = np.zeros(d)
    for filename in names:
        df = pd.read_csv(filename)
        df_url = df[df.To == url]
        N_T = len(df_url)
        if N_T == 0: continue
        for row in df_url.iterrows():
            url = row.From
            if url in top500['url']:
                ind_ancestor = url2ix[url]
                col_ind[ind_ancestor] += 1
        col_ind /= N_T
    col_ind /= len(names)
    if ind < 10:
        ind_str = '00' + str(ind)
    elif ind < 100:
        ind_str = '0' + str(ind)
    else:
        ind_str = str(ind)
    f = gzip.open('column_'+ind_str+'.pkl.gz','wb')
    pickle.dump(col_ind,f,protocol=2)
    f.close()

if __name__ == '__main__':

    indices = np.arange(500,dtype=int)

    pool = Pool(processes=len(names))
    pool.map(worker,names)

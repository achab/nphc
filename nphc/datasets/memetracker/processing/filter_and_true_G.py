from multiprocessing import Pool
from itertools import product
import pandas as pd
import numpy as np
import gzip, pickle
import true_G
import glob


def ix2str(ix):
    if ix < 10:
        ix_str = '00' + str(ix)
    elif ix < 100:
        ix_str = '0' + str(ix)
    else:
        ix_str = str(ix)
    return ix_str

def filter_df(filename):
    df = pd.read_csv(filename)
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df = df[df.index.weekday < 5]
    df = df[(df.index.hour > 13) & (df.index.hour < 23)]
    a = filename.find('/df')
    dir_name = filename[:(a+1)]
    df_name = filename[(a+1):]
    df.to_csv(dir_name + '/filtered_' + df_name, index=False)


if __name__ == '__main__':

    d = 50
    list_df = glob.glob('../data/df*')

    pool1 = Pool()
    pool1.map(filter_df, list_df)

    L = glob.glob('../data/filtered*')
    L.sort()
    idx = 3
    filename = L[idx]
    new_list_df = [filename]

    a = filename.find('20')
    year = filename[a:(a+4)]
    month = filename[(a+5):(a+7)]
    dir_name = '../top{}_1months_start_{}-{}'.format(d, year, month)

    top_d = pd.read_csv('{}/top_{}.csv'.format(dir_name, d))
    ix2url = { ix:url for ix, url in enumerate(top_d['url']) }

    def worker(x):
        return true_G.worker(x,new_list_df,ix2url)
    pool2 = Pool()
    tuple_indices = list(product(range(d),repeat=2))
    res = pool2.map(worker, tuple_indices)

    # save the results
    res_mat = np.array(res).reshape(d,d)
    f = gzip.open(dir_name+'/filtered_true_G.pkl.gz','wb')
    pickle.dump(res_mat,f,protocol=2)
    f.close()

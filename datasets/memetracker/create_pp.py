from multiprocessing import Pool
import pandas as pd
import numpy as np
from glob import glob
import gzip, pickle

top500 = pd.read_csv('top500.csv')

url2ix = { x:i for i, x in enumerate(top500['url']) }
ix2url = { i:x for i, x in enumerate(top500['url']) }

names = glob('df_200*')
names.sort()

start = pd.to_datetime('2008-08-01 00:00:00')

####################
# Useful functions #
####################
def apply_inplace(df, field, fun):
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)

def time2delta(start_time, current_time):
    return int((current_time-start_time).total_seconds())

time_from_start = lambda time: time2delta(start,pd.to_datetime(time))

###################
# Function to map #
###################
def worker(ind):
    """
    Record all jumps of process i in a numpy array.
    """
    url = ix2url[ind]
    process = []
    for filename in names:
        df = pd.read_csv(filename)
        df_url = df[df.To == url]
        if len(df_url) == 0: continue
        df_url = apply_inplace(df_url, 'Date', time_from_start)
        process.append(df_url['Date'].values)
    process_arr = np.concatenate(process)
    if ind < 10:
        ind_str = '00' + str(ind)
    elif ind < 100:
        ind_str = '0' + str(ind)
    else:
        ind_str = str(ind)
    f = gzip.open('process_'+ind_str+'.pkl.gz','wb')
    pickle.dump(process_arr,f,protocol=2)
    f.close()


if __name__ == '__main__':

    indices = np.arange(500,dtype=int)

    pool = Pool(processes=20)
    pool.map(worker, indices)
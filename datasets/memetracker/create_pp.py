from multiprocessing import Pool
import pandas as pd
import numpy as np
from glob import glob
import gzip, pickle


####################
# Useful functions #
####################
def apply_inplace(df, field, fun):
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)

def time2delta(start_time, current_time):
    return int((current_time-start_time).total_seconds())


###################
# Function to map #
###################
def worker(ind,list_df,start,ix2url,dir_name):
    """
    Record all jumps of process i in a numpy array.
    """
    time_from_start = lambda time: time2delta(start,pd.to_datetime(time))
    url = ix2url[ind]
    process = []
    for filename in list_df :
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
    f = gzip.open(dir_name+'/process_'+ind_str+'.pkl.gz','wb')
    pickle.dump(process_arr,f,protocol=2)
    f.close()

def main(list_df,d,dir_name,start):

    top_d = pd.read_csv(dir_name + '/top_' + str(d) + '.csv',index=False)

    ix2url = { i:x for i, x in enumerate(top_d['url']) }

    worker_ = lambda x: worker(x,list_df,start,ix2url,dir_name)

    indices = np.arange(d,dtype=int)

    pool = Pool(processes=20)
    pool.map(worker_, indices)
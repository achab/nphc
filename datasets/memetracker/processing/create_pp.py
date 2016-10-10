import pandas as pd
import numpy as np
import gzip, pickle


####################
# Useful functions #
####################
def apply_inplace(df, field, fun):
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)

def time2delta(start_time, current_time):
    return int((current_time-start_time).total_seconds())

def ix2str(ix):
    if ix < 10:
        ix_str = '00' + str(ix)
    elif ix < 100:
        ix_str = '0' + str(ix)
    else:
        ix_str = str(ix)
    return ix_str

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
        post_nb = df.PostNb.values
        tmp = post_nb[1:] - post_nb[:-1]
        tmp = np.insert(tmp, 0, 1)
        idx_to_keep = np.arange(len(tmp))[tmp == 1]
        df = df.iloc[idx_to_keep]
        df_url = df[df.Blog == url]
        if len(df_url) == 0: continue
        df_url = apply_inplace(df_url, 'Date', time_from_start)
        process.append(df_url['Date'].values)
    if process is not None and len(process):
        process_arr = np.concatenate(process)
        ind_str = ix2str(ind)
        f = gzip.open(dir_name+'/process_'+ind_str+'.pkl.gz','wb')
        pickle.dump(process_arr,f,protocol=2)
        f.close()

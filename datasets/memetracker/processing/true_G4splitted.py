from multiprocessing import Pool
import pandas as pd
import numpy as np
import gzip, pickle
import glob


def ix2str(ix):
    if ix < 10:
        ix_str = '00' + str(ix)
    elif ix < 100:
        ix_str = '0' + str(ix)
    else:
        ix_str = str(ix)
    return ix_str

def plain_worker(ind,list_df,d,dir_name,ix2url):
    """
    Compute the empirical value of || \Phi || by month using the formula
    \int_0^infty \phi^{ij} = N_T^{i \leftarrow j} / N_T^i
    """
    i,j = ind
    f = gzip.open(dir_name+'/new_process_'+ix2str(i)+'.pkl.gz','r')
    process_i = pickle.load(f,encoding='latin1')
    f.close()
    res = []
    if process_i is not None:
        url_i = ix2url[i]
        url_j = ix2url[j]
        for filename in list_df:
            df = pd.read_csv(filename)
            # only keep business days
            # for the right time range
            df = df.set_index('Date')
            df.index = pd.to_datetime(df.index)
            df = df[df.index.weekday < 5]
            df = df[(df.index.hour > 13) & (df.index.hour < 23)]
            # count the number of links
            df_j = df[df.Blog == url_j]
            df_i = df[df.Blog == url_i]
            if len(df_i) ==0: continue
            N_j = df_j.WeightOfLink.sum()
            df_i_from_j = df_i[df_i.Hyperlink == url_j]
            N_i_from_j = df_i_from_j.WeightOfLink.sum()
            try:
                res.append(N_i_from_j / N_j)
            except:
                print(url_j)
        return np.mean(res)
    else:
        return 0.

if __name__ == '__main__':

    d = 200
    dir_name = 'top'+str(d)+'_9months_start_2008_08'
    list_new_df = ['new_df_2008-11.csv','new_df_2008-12.csv','new_df_2009-01.csv','new_df_2009-02.csv']
    top_d = pd.read_csv(dir_name + '/top_' + str(d) + '.csv')
    ix2url = { i:x for i, x in enumerate(top_d['url']) }

    tuple_indices = []
    for i in range(d):
        for j in range(d):
            tuple_indices.append((i,j))

    def worker(ind):
        return plain_worker(ind, list_new_df, d, dir_name, ix2url)

    pool = Pool()

    # save the results
    res = pool.map(worker,tuple_indices)
    res_mat = np.array(res).reshape(d,d)
    f = gzip.open(dir_name+'/true_G_4_splitted.pkl.gz','wb')
    pickle.dump(res_mat,f,protocol=2)
    f.close()

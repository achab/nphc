from multiprocessing import Pool
import pandas as pd
import numpy as np
import os, gzip, pickle, glob


# Define the parameters
#
# d: the number of most cited sites you want to keep
#
# list_df: the list of months you want to keep
#


d = 20

list_df = glob.glob('new_df*')
list_df.sort()

dir_name = "top" + str(d) + "_" + str(len(list_df)) + "months_start_2008_08"

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

if __name__ == '__main__':

    import count_top, create_pp, true_G

    # counts the occurences of the sites
    def worker1(x):
        return count_top.worker(x,dir_name)
    pool1 = Pool(processes=len(list_df))
    pool1.map(worker1,list_df)

    # save the top d sites
    count_top.save_top_d(d,dir_name)

    # useful variables for the worker below
    #start_month = list_df[0][3:-4]
    start_month = '2008-08'
    start = pd.to_datetime(start_month + '-01 00:00:00')
    top_d = pd.read_csv(dir_name + '/top_' + str(d) + '.csv')
    ix2url = { i:x for i, x in enumerate(top_d['url']) }


    # create multivariate point process for the top d sites
    def worker2(x):
        return create_pp.worker(x,list_df,start,ix2url,dir_name)
    indices = np.arange(d,dtype=int)
    pool2 = Pool()
    pool2.map(worker2,indices)

    # estimate G from the labelled links
    def worker3(x):
        return true_G.worker(x,list_df,d,dir_name,ix2url)
    tuple_indices = []
    for i in range(d):
        for j in range(d):
            tuple_indices.append((i,j))
    pool3 = Pool()

    # save the results
    res = pool3.map(worker3,tuple_indices)
    res_mat = np.array(res).reshape(d,d)
    f = gzip.open(dir_name+'/true_G.pkl.gz','wb')
    pickle.dump(res_mat,f,protocol=2)
    f.close()



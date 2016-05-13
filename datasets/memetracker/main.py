from multiprocessing import Pool
import pandas as pd
import numpy as np
import os

# Define the parameters
#
# d: the number of most cited sites you want to keep
#
# list_df: the list of months you want to keep
#


d = 10

list_df = ['df_2008-08.csv']
list_df.sort()

dir_name = "top" + str(d) + "_" + str(len(list_df)) + "months_start" + list_df[0][3:-4]

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

if __name__ == '__main__':

    import count_top, create_pp

    # counts the occurences of the sites
    worker1 = lambda x: count_top.worker(x,dir_name)
    pool1 = Pool(processes=len(list_df))
    pool1.map(worker1,list_df)

    # save the top d sites
    count_top.save_top_d(d,dir_name)

    # useful variables for the worker below
    start_month = list_df[0][3:-4]
    start = pd.to_datetime(start_month + '-01 00:00:00')
    top_d = pd.read_csv(dir_name + '/top_' + str(d) + '.csv')
    ix2url = { i:x for i, x in enumerate(top_d['url']) }


    # create multivariate point process for the top d sites
    worker2 = lambda x: create_pp.worker(x,list_df,start,ix2url,dir_name)
    indices = np.arange(d,dtype=int)
    pool2 = Pool(processes=20)
    pool2.map(worker2,indices)



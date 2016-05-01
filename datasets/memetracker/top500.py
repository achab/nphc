from multiprocessing import Pool
import pandas as pd
import numpy as np

def worker(name):
    """
    Return unique and counts
    """
    # Find an available filename
    filename = 'output_00.csv'
    from os.path import isfile
    while isfile(filename):
        i = int(filename[7:9])
        if i < 10:
            filename = 'output_0{}.csv'.format(i)
        else:
            filename = 'output_{}.csv'.format(i)
    # Count the websites
    X = pd.read_csv(name).values
    posts = np.unique(X[:,1],return_counts=True)
    links = np.unique(X[:,3],return_counts=True)
    # Save the result
    import gzip, pickle
    f = gzip.open(filename, 'wb')
    pickle.dump([posts,links], f, protocol=2)
    f.close()


if __name__ == "__main__":
    import glob
    names = glob.glob("df_200*")

    pool = Pool(processes=15)
    pool.map(worker, names)
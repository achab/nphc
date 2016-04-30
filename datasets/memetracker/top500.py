from multiprocessing import Pool
import pandas as pd

chunksize = 50

data = pd.read_csv("big_data.csv", chunksize=chunksize)
output = []

def worker(some_data):
    print(len(some_data))

if __name__ == "__main__":

    pool = Pool(chunksize)
    pool.map(worker, some_data)

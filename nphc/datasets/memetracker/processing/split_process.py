from multiprocessing import Pool
import pandas as pd
import datetime
import gzip, pickle
import glob

dir_name = 'top150_7months_start_2008-10'

def date_from_start(x):
    return datetime.datetime(2008,10,1) + datetime.timedelta(seconds=int(x))

def worker(filename):
    """
    Input: a whole process
    Output: a list of processes for each day
    """
    f = gzip.open(filename,'r')
    process = pickle.load(f,encoding='latin1')
    f.close()
    df = pd.DataFrame(process,index=list(map(date_from_start,process)))
    if len(df) > 0:
        res = []
        for group in df.groupby([df.index.year,df.index.month,df.index.day]):
            if group[1].index[0].weekday() < 5:
                one_day = group[1][ (group[1].index.hour > 13) & (group[1].index.hour < 23)].values
                res.append(one_day[:,0])
        f = gzip.open(filename[:-7]+'_splitted.pkl.gz','wb')
        pickle.dump(res,f,protocol=2)
        f.close()


if __name__ == "__main__":

    names = glob.glob(dir_name + '/new_process*')
    pool = Pool()
    pool.map(worker,names)

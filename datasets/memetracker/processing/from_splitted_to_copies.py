import pandas as pd
import datetime
import gzip, pickle
import glob

d = 50
year = 2008
month = 11

dir_name = 'top{}_1months_start_{}-{}'.format(d, year, month)

idx_to_keep = []

L = glob.glob(dir_name + '/*splitted*')
L.sort()

def ix2str(ix):
    if ix < 10:
        ix_str = '00' + str(ix)
    elif ix < 100:
        ix_str = '0' + str(ix)
    else:
        ix_str = str(ix)
    return ix_str

tmp = []

for i in idx_to_keep:
    filename = L[i]
    f = gzip.open(filename, 'r')
    process = pickle.load(f, encoding='latin1')
    f.close()
    tmp.append(process)

for day in range(len(res[0])):
    res = []
    for proc in range(len(res)):
        res.append(tmp[proc][day])
    f = gzip.open(dir_name+'/copy_'+ix2str(day)+'.pkl.gz', 'wb')
    pickle.dump(copy, f, protocol=2)
    f.close()

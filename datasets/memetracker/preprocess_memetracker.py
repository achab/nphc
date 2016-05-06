import pandas as pd
import gzip
try:
    # Python 2
    from urlparse import urlparse
except:
    # Python 3
    from urllib.parse import urlparse


def raw2df(filename):
    with gzip.open(filename, 'r') as f:
        content = [x.decode().strip('\n') for x in f.readlines()]
        df_rows = []
        for line in content:
            x = line.split('\t')
            if x[0] == 'P':
                post_url = x[1]
            elif x[0] == 'T':
                date = x[1]
            elif x[0] == 'L':
                hyperlink = x[1]
                row = [post_url, date, hyperlink]
                df_rows.append(row)
        df = pd.DataFrame(df_rows, columns=['Post_URL', 'Date', 'HyperLink'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.to_csv("df_"+filename[7:14]+".csv")

def apply_inplace(df, field, fun):
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)

def parse_url(url):
    o = urlparse(url)
    return o.scheme + "://" + o.netloc

def save_in_one_file():
    import glob
    import os
    list_files = glob.glob('quotes*')
    path = 'file.h5'
    if os.path.exists(path):
        os.remove(path)
    with pd.get_store(path) as store:
        for f in list_files:
            df = pd.read_csv(f)
            store.append('df',df[['url','count']])


"""
if __name__ == '__main__':
    import gzip
    import glob
    names = glob.glob("quotes*")

    pool = Pool(processes=9)
    pool.map(file2df, names)
"""

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import os

    files = ['test1.csv','test2.csv']
    for f in files:
        pd.DataFrame(np.random.randn(10,2),columns=list('AB')).to_csv(f)

    path = 'test.h5'
    if os.path.exists(path):
        os.remove(path)

    with pd.get_store(path) as store:
        for f in files:
            df = pd.read_csv(f,index_col=0)
            try:
                nrows = store.get_storer('foo').nrows
            except:
                nrows = 0

            df.index = pd.Series(df.index) + nrows
            store.append('foo',df)

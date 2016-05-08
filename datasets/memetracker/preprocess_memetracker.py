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
        #content = [x.decode().strip('\n') for x in f.readlines()]
        df_rows = []
        for line in f:
        #for line in content:
            x = line.rstrip('\n').split('\t')
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
        df = df[['Post_URL', 'Date', 'HyperLink']]
        df = apply_inplace(df, 'Post_URL', parse_url)
        df = apply_inplace(df, 'HyperLink', parse_url)
#        df.to_csv("df_"+filename[7:14]+".csv")
    return df

def apply_inplace(df, field, fun):
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)

def parse_url(url):
    try:
        o = urlparse(url)
        return o.scheme + "://" + o.netloc
    except ValueError:
        print("pb with url : ",url)
        return url

def count_links_and_posts():
    hdf = pd.read_hdf('store.h5')
    posts = hdf['Post_URL'].value_counts()
    links = hdf['HyperLink'].value_counts()
    return posts, links


def save_in_one_file():
    import glob
    import os
    list_files = glob.glob('quotes*')
    path = 'store.h5'
    if os.path.exists(path):
        os.remove(path)
    with pd.get_store(path) as store:
        for filename in list_files:
            df = raw2df(filename)
            store.append('df',df)


"""
if __name__ == '__main__':
    import gzip
    import glob
    names = glob.glob("quotes*")

    pool = Pool(processes=9)
    pool.map(file2df, names)
"""

if __name__ == '__main__':

    save_in_one_file()

    posts, links = count_links_and_posts()
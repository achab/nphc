from multiprocessing import Pool
import pandas as pd
import gzip, glob
try:
    # Python 2
    from urlparse import urlparse
except:
    # Python 3
    from urllib.parse import urlparse

####################
# Useful functions #
####################
def apply_inplace(df, field, fun):
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)

def parse_url(url):
    try:
        o = urlparse(url)
        return o.scheme + "://" + o.netloc
    except ValueError:
        #print("pb with url : ",url)
        return url

###################
# Function to map #
###################
def worker(filename):
    """
    Convert raw file into DataFrame
    """
    with gzip.open(filename, 'rb') as f:
        #content = [x.decode().strip('\n') for x in f.readlines()]
        df_rows = []
        for line in f:
        #for line in content:
            x = line.rstrip('\n').split('\t')
            if x[0] == 'P':
                post_url = x[1]
                links_in_same_post = []
            elif x[0] == 'T':
                date = x[1]
            elif x[0] == 'L':
                hyperlink = x[1]
                row = [date, hyperlink, post_url]
                if hyperlink not in links_in_same_post:
                    links_in_same_post.append(hyperlink)
                    df_rows.append(row)
        df = pd.DataFrame(df_rows, columns=['Date', 'Hyperlink', 'Blog'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = apply_inplace(df, 'Hyperlink', parse_url)
        df = apply_inplace(df, 'Blog', parse_url)
        df.to_csv("df_"+filename[7:14]+".csv",index=False)


if __name__ == "__main__":
    import glob
    names = glob.glob("quotes*")

    pool = Pool(processes=len(names))
    pool.map(worker, names)

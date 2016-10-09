import numpy as np
import pandas as pd

def worker(filename,dir_name):
    """
    Return counts by url for the DataFrame in filename
    """
    data = pd.read_csv(filename)
    post_nb = data.PostNb.values
    tmp = post_nb[1:] - post_nb[:-1]
    tmp = np.insert(tmp, 0, 1)
    idx_to_keep = np.arange(len(tmp))[tmp == 1]
    data = data.iloc[idx_to_keep]
    counts = data.Blog.value_counts()
    counts = pd.concat([pd.DataFrame(counts.index),pd.DataFrame(counts.values)],axis=1)
    counts.columns = ['url','count']
    counts.to_csv(dir_name+"/counts_"+filename[8:],index=False)

def save_top_d(d,dir_name):
    import glob
    counts = glob.glob(dir_name + "/counts_*")
    frames = []
    for filename in counts:
        frames.append(pd.read_csv(filename))
    df = pd.concat(frames)
    df.columns = ['url','count']
    df = df.groupby('url').agg({'count':'sum'})
    df = pd.concat([pd.DataFrame(df.index),pd.DataFrame(df.values)],axis=1)
    df.columns = ['url','count']
    df = df.sort('count',ascending=False)
    df[:d].to_csv(dir_name+"/top_"+str(d)+".csv",index=False)

from multiprocessing import Pool
import pandas as pd

def worker(filename,dir_name):
    """
    Return counts by url for the DataFrame in filename
    """
    data = pd.read_csv(filename)
    posts = data['From']
    links = data['To']
    counts = pd.concat([posts,links]).value_counts()
    counts = pd.concat([pd.DataFrame(counts.index),pd.DataFrame(counts.values)],axis=1)
    counts.columns = ['url','count']
    counts.to_csv(dir_name+"/counts_"+filename[3:],index=False)

def main(list_df,d,dir_name):

    worker_ = lambda x: worker(x,dir_name)

    pool = Pool(processes=len(list_df))
    pool.map(worker_, list_df)

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

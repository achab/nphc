from multiprocessing import Pool
import pandas as pd

def worker(filename):
    """
    Return counts by url for the DataFrame in filename
    """
    data = pd.read_csv(filename)
    posts = data['From']
    links = data['To']
    counts = pd.concat([posts,links]).value_counts()
    counts = pd.concat([pd.DataFrame(counts.index),pd.DataFrame(counts.values)],axis=1)
    counts.columns = ['url','count']
    counts.to_csv("counts_"+filename[3:],index=False)

if __name__ == "__main__":
    import glob
    names = glob.glob("df_*")

    pool = Pool(processes=len(names))
    pool.map(worker, names)

    counts = glob.glob("counts_*")
    frames = []
    for filename in counts:
        frames.append(pd.read_csv(filename))
    df = pd.concat(frames)
    df.columns = ['url','count']
    df = df.groupby('url').agg({'count':'sum'})
    df = pd.concat([pd.DataFrame(df.index),pd.DataFrame(df.values)],axis=1)
    df.columns = ['url','count']
    df = df.sort('count',ascending=False)
    df[:500].to_csv("top500.csv",index=False)

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
    counts.to_csv("counts_"+filename[3:],columns=['url','count'])

if __name__ == "__main__":
    import glob
    names = glob.glob("df_*")

    pool = Pool(processes=len(names))
    pool.map(worker, names)

    #counts = glob.glob("counts_*")
    #frames = []
    #for filename in counts:
    #    frames.append(pd.read_csv(filename))
    #df = pd.concat(frames)
    #df = df.groupby('url').sum()
    #df = df.sort_values(by='count',ascending=False)
    #df[:500].to_csv("top500.csv",header=False)

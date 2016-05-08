from multiprocessing import Pool
import pandas as pd

def worker(filename):
    """
    Return counts by url for the DataFrame in filename
    """
    data = pd.read_csv(filename)
    posts = data['Post_URL']
    links = data['HyperLink']
    counts = posts.add(links).value_counts()
    counts.to_csv("counts_"+filename[3:])

if __name__ == "__main__":
    import glob
    names = glob.glob("df_*")

    pool = Pool(processes=len(names))
    pool.map(worker, names)
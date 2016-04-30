from multiprocessing import Pool
import pandas as pd


def file2df(filename):
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

if __name__ == '__main__':
    import gzip
    import glob
    names = glob.glob("quotes*")

    pool = Pool(processes=9)
    pool.map(file2df, names)

import pandas as pd
import gzip
import glob

L = glob.glob("quotes*")

for filename in L:
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
    print("len(df = ",len(df))
    i = 0
    with gzip.open(filename) as f:
        content = [x.strip('\n') for x in f.readlines()]
        for line in content:
            if line.split('\t')[0] == 'L': i += 1
    print("Number of lines starting with 'L' = ",i)
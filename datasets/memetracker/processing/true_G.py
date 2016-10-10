import pandas as pd
import gzip, pickle


def ix2str(ix):
    if ix < 10:
        ix_str = '00' + str(ix)
    elif ix < 100:
        ix_str = '0' + str(ix)
    else:
        ix_str = str(ix)
    return ix_str

def worker(ind,list_df,d,dir_name,ix2url):
    """
    Compute the empirical value of || \Phi || by month using the formula
    \int_0^infty \phi^{ij} = N_T^{i \leftarrow j} / N_T^i
    """
    i,j = ind
    f = gzip.open(dir_name+'/process_'+ix2str(i)+'.pkl.gz','r')
    process_i = pickle.load(f,encoding='latin1')
    f.close()
    if process_i is not None:
        url_i = ix2url[i]
        url_j = ix2url[j]
        res = 0
        for filename in list_df:
            data = pd.read_csv(filename)
            df_j = df[df.Blog == url_j]
            df_i =  df[df.Blog == url_i]
            if len(df_i) == 0: continue
            N_j = df_j.WeightOfLink.sum()
            df_i_from_j = df_i[df_i.Hyperlink == url_j]
            N_i_from_j = df_from_j_to_i.WeightOfLink.sum()
            res += N_i_from_j / N_j
        res *= 1.0/len(list_df)
        return res
    else:
        return 0



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

def worker(ind,list_df,ix2url):
    """
    Compute the empirical value of || \Phi || by month using the formula
    \int_0^infty \phi^{ij} = N_T^{i \leftarrow j} / N_T^i
    """
    i,j = ind
    url_i = ix2url[i]
    url_j = ix2url[j]
    res = 0
    for filename in list_df:
        df = pd.read_csv(filename)
        df_j = df[df.Blog == url_j]
        df_i =  df[df.Blog == url_i]
        if len(df_i) == 0: continue
        N_j = df_j.WeightOfLink.sum()
        df_i_from_j = df_i[df_i.Hyperlink == url_j]
        N_i_from_j = df_i_from_j.WeightOfLink.sum()
        if N_j > 0: res += N_i_from_j / N_j
    res *= 1.0/len(list_df)
    return res

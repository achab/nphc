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
    N_i = len(process_i)
    url_i = ix2url[i]
    url_j = ix2url[j]
    N_from_j_to_i = 0
    for filename in list_df:
        df = pd.read_csv(filename)
        df_to_i = df[df.To == url_i]
        if len(df_to_i) == 0: continue
        df_from_j_to_i = df_to_i[df_to_i.From == url_j]
        N_from_j_to_i += len(df_from_j_to_i)
    res = N_from_j_to_i/float(N_i)
    return res



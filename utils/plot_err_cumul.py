import numpy as np
import pickle, gzip
from utils.metrics import rel_err
from utils.cumulants import Cumulants

mode = 'rect_d100_nonsym'

def err_plot_cumul(H):
    err_C = []
    err_K = []
    err_K_part = []
    logT_arr = np.arange(5,10)
    for logT in logT_arr:
        filename = '.../datasets/' + mode + '_log10T{}_with_Beta.pkl.gz'.format(str(logT))
        f = gzip.open(filename, 'rb')
        data = pickle.load(f)
        f.close()
        cumul, Beta = data
        # Define new Cumulants instance
        new_cumul = Cumulants(cumul.N,hMax=H)
        new_cumul.set_R_true(cumul.R_true)
        new_cumul.set_C_th()
        new_cumul.set_K_th()
        new_cumul.set_K_part_th()
        new_cumul.compute_all(H)
        new_cumul.compute_B(H)
        new_cumul.set_K_part(H)

        err_C.append(rel_err(new_cumul.C_th,new_cumul.C))
        err_K.append(rel_err(new_cumul.K_th,new_cumul.K))
        err_K_part.append(rel_err(new_cumul.K_part_th,new_cumul.K_part))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    ax.plot(logT_arr,err_C,'r',logT_arr,err_K,'b',logT_arr,err_K_part,'g')
    ax.set_ylim(0.,0.15)
    ax.set_title('hMax = {0}'.format(H))
    ax.set_yticks(list(0.01*np.arange(1,16)), minor=False)
    ax.set_yticks(list(0.005*np.arange(1,31)), minor=True)
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor')
    ax.grid(True)
    fig.savefig('.../datasets/opt_H/' + mode[:4] + '/' + mode + '_H{0}.png'.format(H))
    plt.close(fig)

for h in [10,20,30,40,50,60,70,80,90,100,1000,2000]:
#for h in np.arange(1,21):
    err_plot_cumul(h)
    print("done for h = ",h)
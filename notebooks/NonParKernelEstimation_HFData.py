# coding: utf-8

import pandas as pd
import numpy as np
#from nphc.main import NPHC
#from nphc.utils.cumulants import Cumulants
from scipy.linalg import sqrtm, inv, eigh
from numpy import sqrt

import mlpp.hawkesnoparam as hnp

import tensorflow as tf

def GetComponents(df, vol_bins):
    """ Given df representing one day of data, returns list of array representing
        trades time for each volume component"""

    mask = lambda x,y: (df.OrderType==0).values * (np.abs(df.Volume.values)>x) * (np.abs(df.Volume.values)<=y)

    out = []
    for ii in range(len(vol_bins)-1):

        out += [np.copy(df.loc[mask(vol_bins[ii], vol_bins[ii+1]), 'Time'].values)]

    return out

# ## Data store

# data for Eurex futures are in this store
store = pd.HDFStore('/data/data/QH/L1.h5', 'r')

# each dataframe in the store represents one day
# name of the assets are 'xFDAX' and 'xFGBL'
# to access an asset on a particular day do
# store.get['_ast_name/YYYYmmdd']

# to get one day
df = store.get('xFGBL/20140922')

# in the paper we examined trades and considered multilpe components based on volume
# volume buckets:
bins4D = (0,1,3,10, np.inf)
bins6D = (0, 2, 3, 7, 20, np.inf)

asset = 'xFGBL'
days = pd.bdate_range(start='20140101', end='20140701')
big_data = []

# add realization and estimate claw
# 4D case
for d in days:

    try:
        df = store.get('%s/%s' %(asset, d.strftime('%Y%m%d')))
    except KeyError:
        continue

    data = GetComponents(df, bins4D)
    big_data.append(data)


from nphc.main import starting_point, NPHC
import nphc.utils.cumulants as cu
cumul = cu.Cumulants(big_data)

cumul.set_all(H=1e0)

R0 = starting_point(cumul, random=False)
n_epoch = 10001
l_r = 1e-2
opt = 'adam'
disp_step = 200


R = NPHC(cumul, R0, alpha=.9, training_epochs=n_epoch, learning_rate=l_r, optimizer=opt, display_step=disp_step)

import gzip, pickle
f = gzip.open("res.pkl.gz")
pickle.dump(R,f)
f.close()

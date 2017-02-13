from nphc.cumulants import Cumulants
from nphc.main import NPHC
import numpy as np
import mlpp.simulation as hk


#####################################################
### Simulation of a 10-dimensional Hawkes process ###
#####################################################
beta = 1.
mu = 0.01
d = 10
T = 1e6
H = 10

mus = mu * np.ones(d)
Alpha = np.zeros((d,d))
Beta = np.zeros((d,d))
for i in range(5):
    for j in range(5):
        if i <= j:
            Alpha[i][j] = 1.
            Beta[i][j] = 100*beta
for i in range(5,10):
    for j in range(5,10):
        if i >= j:
            Alpha[i][j] = 1.
            Beta[i][j] = beta
Alpha /= 6

kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)] for (a_list, b_list) in zip(Alpha, Beta)]
h = hk.SimuHawkes(kernels=kernels, baseline=list(mus), end_time=T)
h.simulate()


######################################
### Fit (=> compute the cumulants) ###
######################################
nphc = NPHC()
nphc.fit(h.timestamps,half_wifth=10,filter="rectangular",mu_true=mus,R_true=inv(np.eye(d)-Alpha))
# print mean error of cumulants estimation
from nphc.utils.metrics import rel_err
print("mean rel_err on L = ", np.mean( [rel_err(nphc.L_th, L) for L in nphc.L] ))
print("mean rel_err on C = ", np.mean( [rel_err(nphc.C_th, C) for C in nphc.C] ))
print("mean rel_err on K_c = ", np.mean( [rel_err(nphc.K_c_th, K_c) for K_c in nphc.K_c] ))

#################################################
### Solve (=> minimize the objective function ###
#################################################
R_pred = nphc.solve(alpha=.9,training_epochs=100,display_step=20,learning_rate=1e-1,optimizer='adam')

# print final error of estimation
from scipy.linalg import inv
G_pred = np.eye(d) - inv(R_pred)
print(rel_err(Alpha,G_pred))
from nphc.main import NPHC
import numpy as np
from scipy.linalg import inv
import mlpp.simulation as hk


#####################################################
### Simulation of a 10-dimensional Hawkes process ###
#####################################################

# load financial data


######################################
### Fit (=> compute the cumulants) ###
######################################
nphc = NPHC()
nphc.fit(h.timestamps,half_wifth=10,filter="rectangular",method="classic",mu_true=mus,R_true=inv(np.eye(d)-Alpha))

#################################################
### Solve (=> minimize the objective function ###
#################################################
R_pred = nphc.solve(alpha=.9,training_epochs=300,display_step=20,learning_rate=1e-2,optimizer='adam')

# print final error of estimation
G_pred = np.eye(d) - inv(R_pred)
print(rel_err(Alpha,G_pred))

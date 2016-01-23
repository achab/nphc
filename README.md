# NPHC: NonParametric Hawkes with Cumulants

This repo contains two very different works:

**NPHC2**: framework to minimize a loss on ||\Phi||, with three constraints:
- one between empirical intensities (1^{st} integrated cumulant), empirical integrated covariance (2^{nd} integrated cumulant) and ||\Phi||)
- one to ensure the kernels to be causal (i.e. equals zero when t < 0)
- one to ensure stability of the process (spectral radius of ||\Phi|| < 1)

**NPHC3**: regression on the third integrated cumulant

See notebooks to understand how to use the class *Cumulants* and the optimization and estimation processes

# TODO

Add new solves: AdaGrad, AdaDelta

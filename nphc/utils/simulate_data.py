from itertools import product
import numpy as np


def args2params(mode, symmetric):

    #from math import log
    #beta0 = log(1000) / 40.
    beta0 = 0.01
    mu_d10 = 0.001
    mu_d20 = 0.001
    mu_d100 = 0.001
    mu_d500 = 0.01

    if 'd4' in mode:
        d = 4
        mu = 100 * mu_d10 * np.ones(d)
        Alpha = np.zeros((d,d))
        Beta = np.ones((d,d))
        Gamma = .01 * np.ones((d,d))
        Alpha[:d/2,:d/2] += 1.
        Alpha[:d/2,d/2:] += 2.
        Alpha[d/2:,d/2:] += 3.
        Alpha /= 8.

    elif 'd10_sym' in mode:
        d = 10
        mu = mu_d10 * np.ones(d)
        Alpha = np.zeros((d,d))
        Beta = np.zeros((d,d))
        Alpha[:d/2,:d/2] += 1.
        Alpha[d/2:,d/2:] += 1.
        Beta[:d/2,:d/2] += 1000*beta0
        Beta[d/2:,d/2:] += 10*beta0
        if mode == 'd10_sym_hard':
            Alpha[6:8,:3] += 3.
            Beta[6:8,:3] += 100*beta0
        Alpha = .5*(Alpha+Alpha.T)
        Gamma = .5*Alpha
        Beta = .5*(Beta + Beta.T)
        Alpha /= 12

    elif 'd10_nonsym_1' in mode:
        d = 10
        mu = mu_d10 * np.ones(d)
        Alpha = np.zeros((d,d))
        Beta = np.zeros((d,d))
        for i in range(5):
            for j in range(5):
                if i <= j:
                    Alpha[i][j] = 1.
                    Beta[i][j] = 1000*beta0
        for i in range(5,10):
            for j in range(5,10):
                if i >= j:
                    Alpha[i][j] = 1.
                    Beta[i][j] = 10*beta0
        if mode == 'd10_nonsym_1_hard':
            Alpha[6:8,1:3] += 1
            Beta[6:8,1:3] += 100*beta0
        Gamma = Alpha.copy()
        Alpha /= 6

    elif 'd10_nonsym_2' in mode:
        d = 10
        mu = mu_d10 * np.ones(d)
        Alpha = np.zeros((d,d))
        Gamma = np.zeros((d,d))
        for i in range(5):
            for j in range(5):
                if i <= j:
                    Alpha[i][j] = 1.
                    Gamma[i][j] = 1000*beta0
        for i in range(5,10):
            for j in range(5,10):
                if i >= j:
                    Alpha[i][j] = 1.
                    Gamma[i][j] = 10*beta0
        if mode == 'd10_nonsym_2_hard':
            Alpha[6:8,1:3] += 1
            Gamma[6:8,1:3] += 100*beta0
        Gamma *= .1
        Beta = Alpha.copy()
        Alpha /= 6

    elif mode == 'd20_nonsym_1_hard':
        d = 20
        mu = mu_d20 * np.ones(d)
        Alpha = np.zeros((d,d))
        Beta = np.zeros((d,d))
        for i in range(5):
            for j in range(5):
                if i <= j:
                    Alpha[i][j] = 1.
                    Beta[i][j] = 1000*beta0
        for i in range(5,10):
            for j in range(5,10):
                if i >= j:
                    Alpha[i][j] = 1.
                    Beta[i][j] = 10*beta0
        for i in range(10,15):
            for j in range(15,20):
                Alpha[i][j] = 1.
                Beta[i][j] = 100*beta0
        for i in range(15,20):
            for j in range(10,15):
                Alpha[i][j] = 1.
                Beta[i][j] = 1*beta0
        Gamma = Alpha.copy()
        Alpha /= 6.

    elif mode == 'd20_nonsym_2_hard':
        d = 20
        mu = mu_d20 * np.ones(d)
        Alpha = np.zeros((d,d))
        Gamma = np.zeros((d,d))
        for i in range(5):
            for j in range(5):
                if i <= j:
                    Alpha[i][j] = 1.
                    Gamma[i][j] = 1000*beta0
        for i in range(5,10):
            for j in range(5,10):
                if i >= j:
                    Alpha[i][j] = 1.
                    Gamma[i][j] = 10*beta0
        for i in range(10,15):
            for j in range(15,20):
                Alpha[i][j] = 1.
                Gamma[i][j] = 100*beta0
        for i in range(15,20):
            for j in range(10,15):
                Alpha[i][j] = 1.
                Gamma[i][j] = 1*beta0
        Gamma *= .1
        Beta = Alpha.copy()
        Alpha /= 6.

    elif 'd100_nonsym_1' in mode:
        d = 100
        mu = mu_d100 * np.ones(d)
        Alpha = np.zeros((d,d))
        Beta = np.zeros((d,d))
        for i in range(50):
            for j in range(50):
                if i <= j:
                    Alpha[i][j] = 1.
                    Beta[i][j] = 10*beta0
        for i in range(51,80):
            for j in range(51,80):
                if i >= j:
                    Alpha[i][j] = 1.
                    Beta[i][j] = 100.*beta0
        for i in range(81,100):
            for j in range(81,100):
                if i <= j:
                    Alpha[i][j] = 1.
                    Beta[i][j] = 1000.*beta0
        if mode == 'd100_nonsym_1_hard':
            Alpha[60:80,10:30] += 1
            Beta[60:80,10:30] += 100*beta0
        Gamma = Alpha.copy()
        Alpha /= 40

    elif 'd100_nonsym_2' in mode:
        d = 100
        mu = mu_d100 * np.ones(d)
        Alpha = np.zeros((d,d))
        Gamma = np.zeros((d,d))
        for i in range(50):
            for j in range(50):
                if i <= j:
                    Alpha[i][j] = 1.
                    Gamma[i][j] = 10*beta0
        for i in range(51,80):
            for j in range(51,80):
                if i >= j:
                    Alpha[i][j] = 1.
                    Gamma[i][j] = 100.*beta0
        for i in range(81,100):
            for j in range(81,100):
                if i <= j:
                    Alpha[i][j] = 1.
                    Gamma[i][j] = 1000.*beta0
        if mode == 'd100_nonsym_2_hard':
            Alpha[60:80,10:30] += 1
            Gamma[60:80,10:30] += 100*beta0
        Gamma *= .1
        Beta = Alpha.copy()
        Alpha /= 40

    elif 'd500_nonsym_1' in mode:
        d = 500
        mu = mu_d500 * np.ones(d)
        Alpha = np.zeros((d,d))
        Beta = np.zeros((d,d))
        for (i,j) in product(range(200), repeat=2):
             if i <= j:
                Alpha[i][j] = 1.
                Beta[i][j] = beta0
        for (i,j) in product(range(201,300), repeat=2):
            if i >= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 10*beta0
        for (i,j) in product(range(301,400), repeat=2):
            if i <= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 100*beta0
        for (i,j) in product(range(401,500), repeat=2):
            if i >= j:
                Alpha[i][j] = 1.
                Beta[i][j] = 1000*beta0
        if mode == 'd500_nonsym_1_hard':
            Alpha[301:400] += 1.
            Beta[101:200] += 10000*beta0

    return mu, Alpha, Beta, Gamma


def params2kernels(kernel, Alpha, Beta, Gamma):

    import mlpp.simulation as hk
    from mlpp.base.utils import TimeFunction

    if kernel == 'exp':
        kernels = [[hk.HawkesKernelExp(a, b) for (a, b) in zip(a_list, b_list)] for (a_list, b_list) in zip(Alpha, Beta)]

    elif kernel == 'plaw':
        def kernel_plaw(alpha,beta,gamma,support=-1):
            """
            Alternative definition.
            phi(t) = alpha * beta * gamma / (1 + beta t) ** (1 + gamma)
            """
            if beta > 0:
                return hk.HawkesKernelPowerLaw(alpha*gamma/(beta**gamma),1./beta,1.+gamma,support)
            else:
                return hk.HawkesKernelPowerLaw(0.,1.,1.,support)
        kernels = [[kernel_plaw(a, b, g, -1) for (a, b, g) in zip(a_list, b_list, g_list)] for (a_list, b_list, g_list) in zip(Alpha, Beta, Gamma)]

    elif kernel == 'rect':
        def kernel_rect(alpha, beta, gamma):
            if beta > 0:
                T = np.array([0, gamma, gamma + 1./beta ], dtype=float)
                Y = np.array([0, alpha*beta,0], dtype=float)
                tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterConstRight,dt=0.0001)
                return hk.HawkesKernelTimeFunc(tf)
            else:
                T = np.array([0, 1, 1.5 ], dtype=float)
                Y = np.array([0, 0, 0], dtype=float)
                tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterConstRight)
                return hk.HawkesKernelTimeFunc(tf)
        kernels = [[kernel_rect(a, b, g) for (a, b, g) in zip(a_list, b_list, g_list)] for (a_list, b_list, g_list) in zip(Alpha, Beta, Gamma)]

    return kernels


def simulate_and_compute_cumul(mu, kernels, Alpha, T, hM=20):
    import mlpp.simulation as hk
    h = hk.SimuHawkes(kernels=kernels, baseline=list(mu), end_time=T)
    h.simulate()
    # use the class Cumulants
    from nphc.cumulants import Cumulants
    N = h.timestamps
    cumul = Cumulants(realizations=N,half_width=hM)
    # compute everything
    from scipy.linalg import inv
    d = Alpha.shape[0]
    R_true = inv(np.eye(d)-Alpha)
    cumul.set_R_true(R_true)
    cumul.set_mu_true(mu)
    cumul.compute_cumulants(method="parallel")

    from nphc.utils.metrics import rel_err
    print("rel_err on C = ", np.mean( [rel_err(cumul.C_th, C) for C in cumul.C] ) )
    print("rel_err on K_c = ", np.mean([rel_err(cumul.K_c_th, K_c) for K_c in cumul.K_c]))

    return cumul


def save(cumul, Alpha, Beta, Gamma, kernel, mode, T, with_params=True, without_N=False, suffix=''):

    from math import log10
    import gzip, pickle
    name = kernel + '_' + mode + '_log10T' + str(int(log10(T)))

    # Create folders if they don't exist yet
    dir_name = 'nphc/datasets/' + kernel
    import os
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    if with_params and without_N:
        tmp = cumul.N.copy()
        cumul.N = None
        data = (cumul,Alpha,Beta,Gamma)
        f = gzip.open(dir_name + '/' + name + '_with_params_without_N' + suffix + '.pkl.gz','wb')
        pickle.dump(data, f, protocol=2)
        f.close()
        cumul.N = tmp

    elif with_params and not without_N:
        data = (cumul,Alpha,Beta,Gamma)
        f = gzip.open(dir_name + '/' + name + '_with_params' + suffix + '.pkl.gz','wb')
        pickle.dump(data, f, protocol=2)
        f.close()

    elif not with_params and without_N:
        tmp = cumul.N.copy()
        cumul.N = None
        f = gzip.open(dir_name + '/' + name + '_without_N' + suffix + '.pkl.gz','wb')
        pickle.dump(cumul, f, protocol=2)
        f.close()
        cumul.N = tmp

    elif not with_params and not without_N:
        f = gzip.open(dir_name + '/' + name + suffix + '.pkl.gz','wb')
        pickle.dump(cumul, f, protocol=2)
        f.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-k",help="Choose a kernel among: 'exp', 'rect' or 'plaw'.",type=str,choices=['exp','rect','plaw'])
    parser.add_argument("-d",help="Choose the dimension of the process: 10 or 100.",type=int,choices=[10,100])
    parser.add_argument("-s",help="Choose the nonsymmetric matrices Alpha, Beta and Gamma you want to simulate the process from.",type=int,choices=[0,1,2,3])
    parser.add_argument("-t",help="log_10 of the length of the simulation ie '3' gives T=1000",type=int,choices=[3,4,5,6,7,8,9,10])
    args = parser.parse_args()


    ## Parse arguments

    if args.k is None:
        kernel = 'exp'
    else:
        kernel = args.k

    if args.d is None:
        d = 10
    else:
        d = args.d

    if args.s is None:
        symmetric = 1
    else:
        symmetric = args.s

    if args.t is None:
        T = 1e5
    else:
        T = 10**args.t

    if symmetric == 0:
        mode = 'd' + str(d) + '_nonsym_1'
    elif symmetric == 1:
        mode = 'd' + str(d) + '_nonsym_2'
    elif symmetric == 2:
        mode = 'd' + str(d) + '_sym'
    elif symmetric == 3:
        mode = 'd' + str(d) + '_sym_hard'

    mu, Alpha, Beta, Gamma = args2params(mode, symmetric)

    kernels = params2kernels(kernel, Alpha, Beta, Gamma)

    cumul = simulate_and_compute_cumul(mu, kernels, Alpha, T, 20)

    save(cumul, Alpha, Beta, Gamma, kernel, mode, T)

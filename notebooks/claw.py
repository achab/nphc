import numpy as np
from numba import jit

#@jit('f8[:](f8[:],f8[:],f8[:])', nopython=True)
def g_ij(ti, tj, grid):
    """ inputs:
        --------
        ti = array of events time of component i
        tj  = array of events times of component j
        grid = grid of edges for the estimation of g

        output:
        --------
        g_ij conditional law array (dimension of grid - 1)"""

    # number of intervals in the grid
    H = len(grid)-1
    Ni = len(ti)
    Nj = len(tj)

    # output claw initialization
    gg = np.zeros(H)

    # end point of the each interval
    end_slice = grid[1:]

    # maximum lag
    lagMax = end_slice[-1]

    # average intensity of ti
    lambda_i = Ni/(ti[-1]-ti[0])

    # counts the tj effectively used
    count = 0

    #index for the i events
    m = 0

    # loop on the tj events
    for k in range(Nj):

        # prevent to go outside range of ti
        if (tj[k] + lagMax >= ti[-1]):
            break

        # index of first ti after tj[k]
        while (m < Ni) and (ti[m] < tj[k]):
            m += 1

        if (m >= Ni):
            break

        # increment number of tj used
        count += 1

        #
        ti_index_lag_delta = m

        # loop on the lags
        for h in range(H):

            # current lag (start)
            lag = grid[h]

            ti_index_lag = ti_index_lag_delta

            # first ti such that ti - tj[k] > lag
            while (ti[ti_index_lag] <= tj[k] + lag):

                ti_index_lag += 1

            # last ti
            while (ti[ti_index_lag_delta] <= tj[k] + end_slice[h]):

                ti_index_lag_delta += 1

            gg[h] += ti_index_lag_delta - ti_index_lag

    # now average and subtract lambda
    for h in range(H):
        gg[h] /= count
        gg[h] /= (end_slice[h]-grid[h])
        gg[h] -= lambda_i

    return gg

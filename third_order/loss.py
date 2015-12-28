import numpy as np
from numba import double, int_, jit

def full_tensor(R, L, C):
    d = len(L)
    F = np.zeros((d,d,d))
    F_1 = np.einsum('im,jm,km->ijk',C,R,R)
    F += F_1
    F += np.einsum('jki',F_1)
    F += np.einsum('kij',F_1)
    F -= 2*np.einsum('m,im,jm,km',L,R,R,R)
    F += np.einsum('im,jm,km->ijk',R,R,C)
    F += np.einsum('im,jm,km->ijk',R,C,R)
    F += np.einsum('im,jm,km->ijk',C,R,R)
    return F

def partial_tensor(R, L, C, ind=3):
    # if ind == 1: j = k
    # elif ind == 2: i = k
    # elif ind == 3: i = j
    return 0

def full_loss(R, L, C, K):
    F = full_tensor(R, L, C)
    d = F.shape[0]
    return .5 * np.sum((F - K)**2) / (d**3)

def partial_loss(R, L, C, K, ind=3):
    G = partial_tensor(R, L, C, ind=ind)
    return .5 * np.sum((G - K)**2) / (d**2)

if __name__ == "__main__":
    
    def full_tensor(R, L, C):
        d = len(L)
        F_1 = np.einsum('im,jm,km->ijk',C,R,R)
        F = np.zeros((d,d,d))
        F += F_1
        F += np.einsum('jki',F_1)
        F += np.einsum('kij',F_1)
        F -= 2*np.einsum('m,im,jm,km',L,R,R,R)
        return F

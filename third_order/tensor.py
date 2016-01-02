import numpy as np
from numba import double, int_, jit

def full_tensor(R, L, C):
    d = len(L)
    K = np.zeros((d,d,d))
    K_1 = np.einsum('im,jm,km->ijk',C,R,R)
    K += K_1
    K += np.einsum('jki',K_1)
    K += np.einsum('kij',K_1)
    K -= 2*np.einsum('m,im,jm,km',L,R,R,R)
    return K

def partial_tensor(R, L, C):
    d = len(L)
    K_partial = np.dot(R**2,C.T)
    K_partial += 2*np.dot(R*(C - np.einsum('ij,j->ij',R,L)),R.T)
    return K_partial

def full_loss(R, L, C, K_emp):
    K = full_tensor(R, L, C)
    d = K.shape[0]
    return .5 * np.sum((K - K_emp)**2) / (d**3)

def partial_loss(R, L, C, K_part_emp):
    d = len(L)
    K_partial = partial_tensor(R, L, C)
    return .5 * np.sum((K_partial - K_part_emp)**2) / (d**2)

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

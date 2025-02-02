import numpy as np
import scipy.linalg as sp

def Hadamard_Transform(c, l, seed=42069):
    """
    Generate Hadamard matrix H and random matrix R
    - c: number of columns
    - l: number of rows
    - seed: random seed
    - return: HR = H*R
    """
    H = sp.hadamard(c)
    np.random.seed(seed)
    R = np.eye(c)[np.random.randint(0, c, l)]
    HR = np.dot(H, R.T)
    del H, R
    return HR

def Gaussian_Sketching(c, l, seed):
    """
    Generate Gaussian Sketching matrix
    - c: number of columns
    - l: number of rows
    - seed: random seed
    - return: omega_i
    """
    np.random.seed(seed)
    omega_i = 1/np.sqrt(l) * (np.random.randn(c,l)).T
    return omega_i

def SRHT_Sketching(c, l, seed, HR=None):
    """
    Generate SRHT Sketching matrix
    - c: number of columns
    - l: number of rows
    - seed: random seed
    - HR: Hadamard matrix
    - return: omega_i
    """
    if HR is None:
        HR = Hadamard_Transform(c, l)
    np.random.seed(seed)
    D_i_left = np.diag(np.sign(np.random.randn(c)))
    D_i_right = np.diag(np.sign(np.random.randn(l)))
    omega_i = np.sqrt(c/l) * np.dot(D_i_left, np.dot(HR, D_i_right))
    return omega_i


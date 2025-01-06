import scipy.io as sp
import numpy as np
import pickle

def RBF(x, y, c):
    
    return np.exp(-(1/c**2)*np.linalg.norm(x-y)**2)

n = 2**14
c = 1000

dataset = sp.loadmat('mnist.mat')

data = np.array(dataset['Z'])

data = data[:n,:]

A = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        A[i,j] = RBF(data[i], data[j], c)

del dataset
del data

filename = 'A_MNIST_16000.pkl'

with open(filename, 'wb') as f:
    pickle.dump(A, f)

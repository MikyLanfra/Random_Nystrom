import numpy as np
import scipy as sp
import pandas as pd
import pickle, bz2, json, os

# Load configuration
config = json.load(open('Utils/config.json', 'r'))
input_folder = config["matrix_input_path"]
output_folder = config["matrix_output_path"]
mnist_filename = config["mnist_filename"]
msd_filename = config["msd_filename"]

if output_folder not in os.listdir():
    os.mkdir(output_folder)

def dump_matrix(filename, A):
    """
    Store a matrix in a file using pickle
    - filename: name of the file
    - A: matrix to store
    - return: None
    """
    with open(filename, 'wb') as f:
        pickle.dump(A, f)

def PolyDecDiagonal(R,n,p):
    """
    Generate a polynomial decaying diagonal matrix
    - R: Effective Rank, number of 1s in the diagonal
    - n: Size of the matrix
    - p: Rate of polynomial decay
    - return: Diagonal matrix
    """
    D_0 = np.ones(R)
    D_1 = np.arange(2, n-R+2)**(-p)
    D = np.concatenate((D_0, D_1))
    return D

def ExpDecDiagonal(R,n,q):
    """
    Generate an exponential decaying diagonal matrix
    - R: Effective Rank, number of 1s in the diagonal
    - n: Size of the matrix
    - q: Rate of exponential decay
    - return: Diagonal matrix
    """
    D_0 = np.ones(R)    
    D_1 = np.exp(-q*np.arange(2, n-R+2))
    D = np.concatenate((D_0, D_1))
    return D

def RBF(x, y, c):
    """
    Radial Basis Function Kernel
    - x: First vector
    - y: Second vector
    - c: Scale parameter
    - return: RBF kernel value
    """
    return np.exp(-(1/c**2)*np.linalg.norm(x-y)**2)

def MNIST(n, c):
    """
    Generate a RBF kernel matrix using the MNIST dataset
    - n: Number of samples
    - c: Scale parameter of the RBF kernel
    - return: RBF kernel matrix
    """
    filepath = input_folder+"/"+mnist_filename
    dataset = sp.io.loadmat(filepath)
    data = np.array(dataset['Z'])
    data = data[:n,:]
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = RBF(data[i], data[j], c)
    del dataset
    del data
    return A

def MSD(n, c):
    """
    Generate a RBF kernel matrix using the MSD dataset
    - n: Number of samples
    - c: Scale parameter of the RBF kernel
    - return: RBF kernel matrix
    """
    filepath = input_folder+"/"+msd_filename
    with bz2.BZ2File(filepath, 'rb') as f:
        lines = f.readlines()

    targets, features = [], []
    for line in lines:
        decoded_line = line.decode('utf-8').strip()
        parts = decoded_line.split()
        
        target = int(parts[0])
        targets.append(target)
        
        feature_values = [float(feat.split(':')[1]) for feat in parts[1:]]
        features.append(feature_values)

    X = pd.DataFrame(features) 
    y = pd.Series(targets, name="Year")

    data = X.to_numpy()
    data = data[:n, :]
    del X, y

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i,j] = RBF(data[i], data[j], c)
    del data
    return A

if __name__ == "__main__":

    A_Poly = PolyDecDiagonal(5,2**13,0.5)
    dump_matrix(output_folder+"/A_Poly.pkl", A_Poly)
    del A_Poly
    print("Polynomial Decay Matrix done")

    A_Exp = ExpDecDiagonal(5,2**13,0.1)
    dump_matrix(output_folder+"/A_Exp.pkl", A_Exp)
    del A_Exp
    print("Exponential Decay Matrix done")

    A_MNIST = MNIST(2**13, 1000)
    dump_matrix(output_folder+"/A_MNIST.pkl", A_MNIST)
    del A_MNIST
    print("MNIST Matrix done")

    A_MSD = MSD(2**13, 1e4)
    dump_matrix(output_folder+"/A_MSD_104.pkl", A_MSD)
    del A_MSD
    print("MSD Matrix with c=1e4 done")

    A_MSD = MSD(2**13, 1e5)
    dump_matrix(output_folder+"/A_MSD_105.pkl", A_MSD)
    del A_MSD
    print("MSD Matrix with c=1e5 done")
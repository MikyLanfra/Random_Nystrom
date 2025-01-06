import scipy as sp
import numpy as np
import pandas as pd
import bz2

def RBF(x, y, c):
    
    return np.exp(-(1/c**2)*np.linalg.norm(x-y)**2)

# Specify the file path to the .bz2 file
file_path = 'YearPredictionMSD.bz2'

# Open and read the bz2 file
with bz2.BZ2File(file_path, 'rb') as f:
    # Read all lines and decode them as text
    lines = f.readlines()

# Parse the data
targets = []  # To store target values (years)
features = []  # To store feature arrays

for line in lines:
    decoded_line = line.decode('utf-8').strip()
    parts = decoded_line.split()
    
    # Extract the target value (year)
    target = int(parts[0])
    targets.append(target)
    
    # Extract features (ignoring the indices)
    feature_values = [float(feat.split(':')[1]) for feat in parts[1:]]
    features.append(feature_values)

# Convert to pandas DataFrame
X = pd.DataFrame(features)  # Features as DataFrame
y = pd.Series(targets, name="Year")  # Target as Series

n = 2**14
c1 = 1e4

data = X.to_numpy()
data = data[:n, :]

del X, y


A1 = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        A1[i,j] = RBF(data[i], data[j], c1)

del data

filename1 = 'A_MSD_104_16000.pkl'

import pickle

with open(filename1, 'wb') as f:
    pickle.dump(A1, f)

del A1


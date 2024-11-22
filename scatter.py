import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n = 27
c = n//np.sqrt(size).astype(int)
A_reshaped = None
if rank == 0:
    A = np.arange(1,n**2+1).reshape(n,n)
    A_list = []
    for i in range(np.sqrt(size).astype(int)):
        A_list.append(A[i*c:(i+1)*c,:])
    A_reshaped = np.hstack(A_list).T
    print(c)
    print(f'Rank {rank} has\n{A_reshaped.shape}')

A_i = np.empty((1, c*c), dtype=np.float64)
comm.Scatterv(A_reshaped, A_i, root=0)
# A_i = A_i.T
# print(f'Rank {rank} has\n{A_i}')

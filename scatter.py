import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n = 9
c = n//np.sqrt(size).astype(int)
matrix_to_send = None
if rank == 0:
    A = np.arange(1,n**2+1).reshape(n,n)
    print(f'Rank {rank} has\n{A}')
    A_list = []
    for i in range(np.sqrt(size).astype(int)):
        A_list.append(A[i*c:(i+1)*c,:])
    A_reshaped = (np.hstack(A_list)).astype(np.float64)
    arrs = np.split(A_reshaped, size, axis=1)
    raveled = [np.ravel(arr) for arr in arrs]
    matrix_to_send = np.concatenate(raveled)

A_i = np.empty((c, c), dtype=np.float64)
comm.Scatterv(matrix_to_send, A_i, root=0)
print(f'Rank {rank} has\n{A_i}')

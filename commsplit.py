import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 8
n_blocks = np.sqrt(size)
A = np.empty((0,0), dtype=np.float64)

comm_cols = comm.Split(color=rank // n_blocks, key=rank % n_blocks)
comm_rows = comm.Split(color=rank % n_blocks, key=rank // n_blocks)

rank_col = comm_cols.Get_rank()
rank_row = comm_rows.Get_rank()
print(f'Rank: {rank}, Rank_row: {rank_row}, Rank_col: {rank_col}')


# Split into first column
if rank == 0:
    A = (np.arange(1,n**2+1).reshape(n,n)).astype(np.float64)
    print(f'Rank {rank} has\n{A}')

if rank_col == 0:
    A_i = np.empty((int(n//n_blocks), n), dtype=np.float64)
else:
    A_i = np.empty((0,0), dtype=np.float64)

comm_rows.Scatterv(A, A_i, root=0)

if rank == 0:
    print(f'Rank {rank} has\n{A_i}')

# Split into first row
matrix_to_send = None
if rank_col == 0:
    arrs = np.split(A_i, np.sqrt(size), axis=1)
    # print(arrs)
    raveled = [np.ravel(arr) for arr in arrs]
    # print(raveled)
    matrix_to_send = np.concatenate(raveled)

A_ij = np.empty((int(n//n_blocks), int(n//n_blocks)), dtype=np.float64)
comm_cols.Scatterv(matrix_to_send, A_ij, root=0)
print(f'Rank {rank} has\n{A_ij}')
### cd C:\Users\miche\Documents\GitHub\HPC-Project2
### mpiexec -n 9 python commsplit.py
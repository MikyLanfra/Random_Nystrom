from mpi4py import MPI
import numpy as np
import scipy as sp
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 1024
l = 50
K = 10
n_blocks = np.sqrt(size).astype(int)
c = int(n // n_blocks)
matrix_to_send = None
A = np.empty((0,0), dtype=np.float64)
Uk_hat = np.empty((n, K), dtype=np.float64)
HR = None

comm_cols = comm.Split(color=rank // n_blocks, key=rank % n_blocks)
comm_rows = comm.Split(color=rank % n_blocks, key=rank // n_blocks)

rank_col = comm_cols.Get_rank()
rank_row = comm_rows.Get_rank()

if rank == 0:
    
    with open("A_Poly_test.pkl", "rb") as f:
        A = pickle.load(f)

    H = sp.linalg.hadamard(c)
    np.random.seed(42069)
    R = np.eye(c)[np.random.randint(0, c, l)]

    nNormA = np.sum(A)

    A = np.diag(A)

    HR = np.dot(H, R.T)

    # A = np.arange(1, 65).reshape(8,8).astype(np.float64)

HR = comm.bcast(HR, root=0)

# Split into first column
if rank_col == 0:
    A_i = np.empty((int(n//n_blocks), n), dtype=np.float64)
else:
    A_i = np.empty((0,0), dtype=np.float64)

comm_rows.Scatterv(A, A_i, root=0)

# Split into first row
if rank_col == 0:
    arrs = np.split(A_i, n_blocks, axis=1)
    raveled = [np.ravel(arr) for arr in arrs]
    matrix_to_send = np.concatenate(raveled)

A_ij = np.empty((c, c), dtype=np.float64)
comm_cols.Scatterv(matrix_to_send, A_ij, root=0)

np.random.seed(int(rank // np.sqrt(size)))
D_i_left_L = np.diag(np.sign(np.random.randn(c)))
D_i_right_L = np.diag(np.sign(np.random.randn(l)))
omega_i_left = np.sqrt(c/l) * np.dot(D_i_left_L, np.dot(HR, D_i_right_L))
np.random.seed(int(rank % np.sqrt(size)))
D_i_left_R = np.diag(np.sign(np.random.randn(c)))
D_i_right_R = np.diag(np.sign(np.random.randn(l)))
omega_i_right = np.sqrt(c/l) * np.dot(D_i_left_R, np.dot(HR, D_i_right_R))

C_ij = np.dot(A_ij, omega_i_right)
B_ij = np.dot(omega_i_left.T, C_ij)


# COMPUTE C
C = np.empty((n, l), dtype=np.float64)

C_i = np.empty((c,l), dtype=np.float64)
comm_cols.Reduce(C_ij, C_i, op = MPI.SUM, root = 0)

comm_rows.Gatherv(C_i, C, root=0)

# COMPUTE B REDUNDANDTLY
B = np.empty((l,l), dtype=np.float64)
comm.Allreduce(B_ij, B, op=MPI.SUM)

# RE-SCATTER C
C_loc = np.empty((int(n/size), l), dtype=np.float64)
comm.Scatterv(C, C_loc, root=0)

# EIGENDEMPOSITION OF B
U, lbda, _ = np.linalg.svd(B)
L = np.dot(U, np.diag(np.sqrt(lbda)))

# COMPUTE Z
Z_loc = np.dot(C_loc, np.linalg.inv(L.T))

# QR FACTORIZATION OF Z
Qs = []
Q_loc, R_loc = np.linalg.qr(Z_loc)
Qs.append(Q_loc)

# Reduction tree to compute R
for k in range(int(np.log2(size))):
    if rank % 2**k != 0:
        break

    J = int(2**(k+1)*np.floor(rank/2**(k+1)) + (rank + 2**k) % 2**(k+1))

    if rank > J:
        comm.send(R_loc, dest = J, tag = 11)


    else:
        R_rec = comm.recv(source = J, tag = 11)
        Q_loc, R_loc = np.linalg.qr(np.vstack([R_loc, R_rec]))
        Qs.append(Q_loc)
    

if rank == 0:
    R = R_loc
    sub_Q = np.array(Qs[-1])


# Reduction tree to compute Q
for k in range(int(np.log2(size))-1, -1, -1):
    if rank % 2**k != 0:
        continue

    J = int(2**(k+1)*np.floor(rank/2**(k+1)) + (rank + 2**k) % 2**(k+1))

    if rank < J:
        comm.send(sub_Q[l:, :], dest = J, tag = 878)
        sub_Q = np.dot(np.array(Qs[k]), sub_Q[:l, :])

    else:
        mult = comm.recv(source = J, tag = 878)
        sub_Q = np.dot(np.array(Qs[k]), mult)


# COMPUTE AND BRAODCAST TRUNCATED SVD OF R
if rank == 0:
    U, S, _ = np.linalg.svd(R)
    Uk = U[:, :K]
    Sk = S[:K]
else:
    Uk = None
    Sk = None

Uk = comm.bcast(Uk, root=0)
Sk = comm.bcast(Sk, root=0)

# COMPUTE LOCAL MULTIPLICATION Q*Uk
Uk_hat_loc = np.dot(sub_Q, Uk)

comm.Gatherv(Uk_hat_loc, Uk_hat, root=0)

if rank == 0:
     
    A_Nyst = np.dot(Uk_hat, np.dot(np.diag(Sk**2), Uk_hat.T))

    _, S_Nyst, _ = np.linalg.svd(A - A_Nyst)

    nNorm = np.sum(S_Nyst)

    print(f"Relative error: {nNorm/nNormA}")
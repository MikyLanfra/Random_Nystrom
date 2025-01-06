from mpi4py import MPI
import numpy as np
import scipy.linalg as sp
from scipy.linalg import svd
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 2**13
l = 200
K = 50
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

eps = np.finfo(np.float64).eps

if rank == 0:
    
    with open("A_Exp_test.pkl", "rb") as f:
        A = pickle.load(f)

    eps = np.finfo(np.float64).eps

    for i in range(len(A)):
        if A[i] < eps:
            A[i] = 0


    time1 = MPI.Wtime()

    H = sp.hadamard(c)
    np.random.seed(42069)
    R = np.eye(c)[np.random.randint(0, c, l)]

    A = np.diag(A)

    HR = np.dot(H, R.T)

    print(f"Time to load A and generate HR: {MPI.Wtime() - time1}")

    time2 = MPI.Wtime()

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

if rank==0: time_scatter = MPI.Wtime()

# COMPUTE C
C_ij = np.dot(A_ij, omega_i_right)
C_i = np.empty((c,l), dtype=np.float64)
comm_cols.Reduce(C_ij, C_i, op = MPI.SUM, root = 0)

# COMPUTE B REDUNDANDTLY
B_ij = np.dot(omega_i_left.T, C_ij)
B = np.empty((l,l), dtype=np.float64)
comm.Allreduce(B_ij, B, op=MPI.SUM)


del HR
if rank == 0:
    del H, R, A

if rank==0: time_BC = MPI.Wtime()


if rank_col == 0:
    # SVD OF B
    U, lbda, _ = np.linalg.svd(B)

    mask = lbda > eps
    U = U[:, mask]
    lbda = lbda[mask]

    L = np.dot(U, np.diag(np.sqrt(lbda)))

    # COMPUTE Z
    Z_loc = np.dot(C_i, np.linalg.inv(L.T))

    # QR FACTORIZATION OF Z
    Qs = []
    Q_loc, R_loc = np.linalg.qr(Z_loc)
    Qs.append(Q_loc)

    # Reduction tree to compute R
    for k in range(int(np.log2(np.sqrt(size)))):
        if rank_row % 2**k != 0:
            break

        J = int(2**(k+1)*np.floor(rank_row/2**(k+1)) + (rank_row + 2**k) % 2**(k+1))

        if rank_row > J:
            comm_rows.send(R_loc, dest = J, tag = 11)


        else:
            R_rec = comm_rows.recv(source = J, tag = 11)
            Q_loc, R_loc = np.linalg.qr(np.vstack([R_loc, R_rec]))
            Qs.append(Q_loc)
        

    if rank_row == 0:
        R = R_loc
        sub_Q = np.array(Qs[-1])


    # Reduction tree to compute Q
    for k in range(int(np.log2(np.sqrt(size)))-1, -1, -1):
        if rank_row % 2**k != 0:
            continue

        J = int(2**(k+1)*np.floor(rank_row/2**(k+1)) + (rank_row + 2**k) % 2**(k+1))

        if rank_row < J:
            comm_rows.send(sub_Q[l:, :], dest = J, tag = 878)
            sub_Q = np.dot(np.array(Qs[k]), sub_Q[:l, :])

        else:
            mult = comm_rows.recv(source = J, tag = 878)
            sub_Q = np.dot(np.array(Qs[k]), mult)

    if rank==0: time_QR = MPI.Wtime()

    # COMPUTE AND BRAODCAST TRUNCATED SVD OF R
    if rank_row == 0:
        U, S, _ = np.linalg.svd(R)
        Uk = U[:, :K]
        Sk = S[:K]
    else:
        Uk = None
        Sk = None

    Uk = comm_rows.bcast(Uk, root=0)
    Sk = comm_rows.bcast(Sk, root=0)

    # COMPUTE LOCAL MULTIPLICATION Q*Uk
    Uk_hat_loc = np.dot(sub_Q, Uk)

    comm_rows.Gatherv(Uk_hat_loc, Uk_hat, root=0)

    if rank==0: time_Uk = MPI.Wtime()

if rank == 0:
     
    A_Nyst = np.dot(Uk_hat, np.dot(np.diag(Sk**2), Uk_hat.T))
    print(f"Time to compute Nystrom: {MPI.Wtime() - time2}")

    print(f"Scatter time: {time_scatter - time2}")
    print(f"BC time: {time_BC - time_scatter}")
    print(f"Scatter and BC time: {time_BC - time2}")
    print(f"QR time: {time_QR - time_BC}")
    print(f"Uk time: {time_Uk - time_QR}")
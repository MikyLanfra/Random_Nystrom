from mpi4py import MPI
import numpy as np
import pickle
from sketching import *

def random_nystrom(n, l, K, A_matrix_name, sketching="Gaussian"):
    """
    Randomized Nystrom method
    - n: High Rank Dimension, number of rows of A
    - l: Sketching Dimension, number of rows of omega_i
    - K: Low Rank Dimension, number of columns of Uk_hat
    - A_matrix_name: Name of the file containing the matrix A
    - sketching: Sketching method, either "Gaussian" or "SRHT"
    - return: A_Nyst, the approximation of A
    """
    # Initialization
    time_start = MPI.Wtime()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
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
        with open(A_matrix_name, "rb") as f:
            A = pickle.load(f)
        A[A < eps] = 0
        
    # Split into first column
    if rank_col == 0: A_i = np.ascontiguousarray(np.empty((int(n//n_blocks), n), dtype=np.float64))
    else: A_i = np.ascontiguousarray(np.empty((0,0), dtype=np.float64))
    comm_rows.Scatterv(np.ascontiguousarray(A), A_i, root=0)

    # Split into first row
    if rank_col == 0:
        arrs = np.split(A_i, n_blocks, axis=1)
        raveled = [np.ravel(arr) for arr in arrs]
        matrix_to_send = np.concatenate(raveled)    
    A_ij = np.empty((c, c), dtype=np.float64)
    comm_cols.Scatterv(matrix_to_send, A_ij, root=0)

    time_init = MPI.Wtime()

    # Generate Sketching matrix
    if sketching == "Gaussian":
        omega_i_left = Gaussian_Sketching(c, l, int(rank // np.sqrt(size)))
        omega_i_right = Gaussian_Sketching(c, l, int(rank % np.sqrt(size)))
    
    elif sketching == "SRHT":
        if rank == 0:
            HR = Hadamard_Transform(c, l)
        HR = comm.bcast(HR, root=0)
        omega_i_left = SRHT_Sketching(c, l, int(rank // np.sqrt(size)), HR)
        omega_i_right = SRHT_Sketching(c, l, int(rank % np.sqrt(size)), HR)
        del HR

    else:
        raise ValueError("Invalid sketching method")

    time_sketch = MPI.Wtime()

    # Compute B and C
    C_ij = np.dot(A_ij, omega_i_right)
    C_i = np.empty((c,l), dtype=np.float64)
    comm_cols.Reduce(C_ij, C_i, op = MPI.SUM, root = 0)

    B_ij = np.dot(omega_i_left.T, C_ij)
    B = np.empty((l,l), dtype=np.float64)
    comm.Allreduce(B_ij, B, op=MPI.SUM)

    if rank==0: del A

    time_BC = MPI.Wtime()

    if rank_col == 0:
        # SVD of B and creation of Z
        U, lbda, _ = np.linalg.svd(B)
        U = U[:, lbda > eps]
        lbda = lbda[lbda > eps]
        L = np.dot(U, np.diag(np.sqrt(lbda)))
        Z_loc = np.dot(C_i, np.linalg.inv(L.T))

        # QR factorization of Z
        Qs = []
        Q_loc, R_loc = np.linalg.qr(Z_loc)
        Qs.append(Q_loc)
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

        time_QR = MPI.Wtime()

        # Truncated SVD of R
        if rank_row == 0:
            U, S, _ = np.linalg.svd(R)
            Uk = U[:, :K]
            Sk = S[:K]
        else:
            Uk = None
            Sk = None

        Uk = comm_rows.bcast(Uk, root=0)
        Sk = comm_rows.bcast(Sk, root=0)
        Uk_hat_loc = np.dot(sub_Q, Uk)
        comm_rows.Gatherv(Uk_hat_loc, Uk_hat, root=0)
        time_Uk = MPI.Wtime()

    
    if rank == 0: 
        A_Nyst = np.dot(Uk_hat, np.dot(np.diag(Sk**2), Uk_hat.T))
        print("Time for initialization: ", time_init - time_start)
        print("Time for sketching: ", time_sketch - time_init)
        print("Time for BC: ", time_BC - time_sketch)
        print("Time for QR: ", time_QR - time_BC)
        print("Time for Uk: ", time_Uk - time_QR)
        print("Total time: ", time_Uk - time_start)
        print("Error: ", np.linalg.norm(A - A_Nyst)/np.linalg.norm(A))
        return A_Nyst
    

def random_nystrom_rescatter(n, l, K, A_matrix_name, sketching="Gaussian"):
    """
    Randomized Nystrom method, with rescattering of C on P processes vs sqrt(P) processes
    - n: High Rank Dimension, number of rows of A
    - l: Sketching Dimension, number of rows of omega_i
    - K: Low Rank Dimension, number of columns of Uk_hat
    - A_matrix_name: Name of the file containing the matrix A
    - sketching: Sketching method, either "Gaussian" or "SRHT"
    - return: A_Nyst, the approximation of A
    """
    # Initialization
    time_start = MPI.Wtime()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
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
        with open(A_matrix_name, "rb") as f:
            A = pickle.load(f)
        A[A < eps] = 0
        
    # Split into first column
    if rank_col == 0: A_i = np.ascontiguousarray(np.empty((int(n//n_blocks), n), dtype=np.float64))
    else: A_i = np.ascontiguousarray(np.empty((0,0), dtype=np.float64))
    comm_rows.Scatterv(np.ascontiguousarray(A), A_i, root=0)

    # Split into first row
    if rank_col == 0:
        arrs = np.split(A_i, n_blocks, axis=1)
        raveled = [np.ravel(arr) for arr in arrs]
        matrix_to_send = np.concatenate(raveled)    
    A_ij = np.empty((c, c), dtype=np.float64)
    comm_cols.Scatterv(matrix_to_send, A_ij, root=0)

    time_init = MPI.Wtime()

    # Generate Sketching matrix
    if sketching == "Gaussian":
        omega_i_left = Gaussian_Sketching(c, l, int(rank // np.sqrt(size)))
        omega_i_right = Gaussian_Sketching(c, l, int(rank % np.sqrt(size)))
    
    elif sketching == "SRHT":
        if rank == 0:
            HR = Hadamard_Transform(c, l)
        HR = comm.bcast(HR, root=0)
        omega_i_left = SRHT_Sketching(c, l, int(rank // np.sqrt(size)), HR)
        omega_i_right = SRHT_Sketching(c, l, int(rank % np.sqrt(size)), HR)
        del HR

    else:
        raise ValueError("Invalid sketching method")

    time_sketch = MPI.Wtime()

    # Compute B and C
    C_ij = np.dot(A_ij, omega_i_right)
    C_i = np.empty((c,l), dtype=np.float64)
    comm_cols.Reduce(C_ij, C_i, op = MPI.SUM, root = 0)
    


    B_ij = np.dot(omega_i_left.T, C_ij)
    B = np.empty((l,l), dtype=np.float64)
    comm.Allreduce(B_ij, B, op=MPI.SUM)

    if rank==0: del A

    time_BC = MPI.Wtime()

    # Rescatter C
    C = np.empty((n, l), dtype=np.float64)
    comm_rows.Gatherv(C_i, C, root=0)
    C_loc = np.empty((int(n/size), l), dtype=np.float64)
    comm.Scatterv(C, C_loc, root=0)
    
    time_Rescatter = MPI.Wtime()

    # SVD of B and creation of Z
    U, lbda, _ = np.linalg.svd(B)
    U = U[:, lbda > eps]
    lbda = lbda[lbda > eps]
    L = np.dot(U, np.diag(np.sqrt(lbda)))
    Z_loc = np.dot(C_loc, np.linalg.inv(L.T))

    # QR factorization of Z
    Qs = []
    Q_loc, R_loc = np.linalg.qr(Z_loc)
    Qs.append(Q_loc)
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

    if rank == 0:
        R = R_loc
        sub_Q = np.array(Qs[-1])
    for k in range(int(np.log2(np.sqrt(size)))-1, -1, -1):
        if rank % 2**k != 0:
            continue
        J = int(2**(k+1)*np.floor(rank/2**(k+1)) + (rank + 2**k) % 2**(k+1))
        if rank < J:
            comm.send(sub_Q[l:, :], dest = J, tag = 878)
            sub_Q = np.dot(np.array(Qs[k]), sub_Q[:l, :])
        else:
            mult = comm.recv(source = J, tag = 878)
            sub_Q = np.dot(np.array(Qs[k]), mult)

    time_QR = MPI.Wtime()

    # Truncated SVD of R
    if rank == 0:
        U, S, _ = np.linalg.svd(R)
        Uk = U[:, :K]
        Sk = S[:K]
    else:
        Uk = None
        Sk = None

    Uk = comm.bcast(Uk, root=0)
    Sk = comm.bcast(Sk, root=0)
    Uk_hat_loc = np.dot(sub_Q, Uk)
    comm.Gatherv(Uk_hat_loc, Uk_hat, root=0)
    time_Uk = MPI.Wtime()

    
    if rank == 0: 
        A_Nyst = np.dot(Uk_hat, np.dot(np.diag(Sk**2), Uk_hat.T))
        print("Time for initialization: ", time_init - time_start)
        print("Time for sketching: ", time_sketch - time_init)
        print("Time for BC: ", time_BC - time_sketch)
        print("Time for Rescatter: ", time_Rescatter - time_BC)
        print("Time for QR: ", time_QR - time_Rescatter)
        print("Time for Uk: ", time_Uk - time_QR)
        print("Total time: ", time_Uk - time_start)
        print("Error: ", np.linalg.norm(A - A_Nyst)/np.linalg.norm(A))
        return A_Nyst
    

if __name__=="__main__":
    A_Nyst = random_nystrom(2**13, 100, 50, "A_MNIST.pkl", "Gaussian")
    
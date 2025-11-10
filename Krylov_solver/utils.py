"""
Utility functions for matrix generation and loading.
"""

import numpy as np
import scipy.sparse as sp


def poisson_2d(nx: int, ny: int):
    """
    Build 2D Poisson matrix on a regular grid (nx * ny) with Dirichlet BC.
    Returns SciPy CSR matrix of size (nx*ny, nx*ny).
    
    Parameters
    ----------
    nx : int
        Number of grid points in x-direction
    ny : int
        Number of grid points in y-direction
    
    Returns
    -------
    A : scipy.sparse.csr_matrix
        The sparse Poisson matrix in CSR format
    """
    N = nx * ny
    main_diag = np.ones(N) * 4.0
    off_diag = np.ones(N - 1) * -1.0
    off_diag2 = np.ones(N - nx) * -1.0

    # Mask out connections across row boundaries
    for i in range(1, ny):
        off_diag[i * nx - 1] = 0.0

    diags = [main_diag, off_diag, off_diag, off_diag2, off_diag2]
    offsets = [0, -1, 1, -nx, nx]
    A = sp.diags(diags, offsets, shape=(N, N), format="csr")
    return A


def load_matrix_market(filename: str):
    """
    Load a sparse matrix from a Matrix Market file.
    
    Parameters
    ----------
    filename : str
        Path to the Matrix Market file (.mtx)
    
    Returns
    -------
    A : scipy.sparse.csr_matrix
        The sparse matrix in CSR format
    """
    from scipy.io import mmread
    A = mmread(filename)
    # Convert to CSR format for efficient operations
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = A.tocsr()
    return A


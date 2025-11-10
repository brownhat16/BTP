"""
Preconditioner implementations for CPU and GPU backends.
"""

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cpspla
    from cupyx.scipy.sparse.linalg import LinearOperator as CuLinearOperator
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def cpu_jacobi_precond(A: sp.spmatrix):
    """
    Build a Jacobi (diagonal) preconditioner as a SciPy LinearOperator.

    M ≈ A^{-1}, implemented as x -> D^{-1} x, where D = diag(A).
    
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix
    
    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator
        Jacobi preconditioner
    """
    from scipy.sparse.linalg import LinearOperator

    D = A.diagonal()
    if np.any(D == 0):
        raise ValueError("Zero diagonal entry, cannot build Jacobi preconditioner.")
    Dinv = 1.0 / D

    def matvec(x):
        return Dinv * x

    n = A.shape[0]
    return LinearOperator((n, n), matvec=matvec)


def cpu_ilu_precond(A: sp.spmatrix, drop_tol=0.0, fill_factor=1.0):
    """
    Build an ILU(0) (or ILU with limited fill) preconditioner using scipy.sparse.linalg.spilu.

    Returns a LinearOperator that applies the approximate inverse LU^{-1}.
    
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix
    drop_tol : float, optional
        Drop tolerance for ILU factorization. Default is 0.0.
    fill_factor : float, optional
        Fill factor for ILU factorization. Default is 1.0 (ILU(0)).
    
    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator
        ILU preconditioner
    """
    from scipy.sparse.linalg import LinearOperator, spilu

    # SuperLU prefers CSC format
    A_csc = A.tocsc()
    ilu = spilu(A_csc, drop_tol=drop_tol, fill_factor=fill_factor)

    def matvec(x):
        return ilu.solve(x)

    n = A.shape[0]
    return LinearOperator((n, n), matvec=matvec)


def gpu_jacobi_precond(A_gpu):
    """
    Jacobi preconditioner for a CuPy sparse matrix A_gpu (CSR).
    M^{-1} = diag(A)^{-1} on the GPU.
    
    Parameters
    ----------
    A_gpu : cupyx.scipy.sparse.csr_matrix
        Sparse matrix on GPU
    
    Returns
    -------
    M : cupyx.scipy.sparse.linalg.LinearOperator
        Jacobi preconditioner on GPU
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")

    D = A_gpu.diagonal()
    if cp.any(D == 0):
        raise ValueError("Zero diagonal entry, cannot build Jacobi preconditioner.")
    Dinv = 1.0 / D

    def matvec(x):
        return Dinv * x

    n = A_gpu.shape[0]
    return CuLinearOperator((n, n), matvec=matvec)


def gpu_ilu_precond(A_gpu, drop_tol=0.0, fill_factor=1.0):
    """
    ILU preconditioner for a CuPy sparse CSR matrix using cupyx.scipy.sparse.linalg.spilu.
    
    Parameters
    ----------
    A_gpu : cupyx.scipy.sparse.csr_matrix
        Sparse matrix on GPU
    drop_tol : float, optional
        Drop tolerance for ILU factorization. Default is 0.0.
    fill_factor : float, optional
        Fill factor for ILU factorization. Default is 1.0 (ILU(0)).
    
    Returns
    -------
    M : cupyx.scipy.sparse.linalg.LinearOperator
        ILU preconditioner on GPU
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")

    ilu = cpspla.spilu(A_gpu, drop_tol=drop_tol, fill_factor=fill_factor)

    def matvec(x):
        return ilu.solve(x)

    n = A_gpu.shape[0]
    return CuLinearOperator((n, n), matvec=matvec)


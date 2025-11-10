"""
Core solver implementations for CPU and GPU backends.
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .preconditioners import (
    cpu_jacobi_precond,
    cpu_ilu_precond,
    gpu_jacobi_precond,
    gpu_ilu_precond,
    CUPY_AVAILABLE,
)

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpspla
except ImportError:
    pass


def to_gpu_csr(A: sp.spmatrix):
    """
    Convert a SciPy CSR/CSC/COO matrix to a CuPy CSR matrix.
    
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix
    
    Returns
    -------
    A_gpu : cupyx.scipy.sparse.csr_matrix
        Sparse matrix on GPU
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")
    # Ensure CSR on CPU first for simplicity
    A_csr = A.tocsr()
    data = cp.asarray(A_csr.data)
    indices = cp.asarray(A_csr.indices)
    indptr = cp.asarray(A_csr.indptr)
    return cpsp.csr_matrix((data, indices, indptr), shape=A_csr.shape)


def to_gpu_vec(x: np.ndarray):
    """Convert NumPy array to CuPy array."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")
    return cp.asarray(x)


def to_cpu_vec(x_gpu):
    """Convert CuPy array to NumPy array."""
    if not CUPY_AVAILABLE:
        return x_gpu
    return cp.asnumpy(x_gpu)


def cpu_solve(A: sp.spmatrix,
              b: np.ndarray,
              method: str = "gmres",
              preconditioner: str = "none",
              tol: float = 1e-8,
              maxiter: int = 1000,
              restart: int | None = 50,
              drop_tol: float = 0.0,
              fill_factor: float = 1.0):
    """
    CPU solve wrapper using SciPy GMRES/BiCGStab with optional Jacobi/ILU preconditioning.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix
    b : numpy.ndarray
        Right-hand side vector
    method : str, optional
        Solver method: "gmres" or "bicgstab". Default is "gmres".
    preconditioner : str, optional
        Preconditioner type: "none", "jacobi", or "ilu". Default is "none".
    tol : float, optional
        Convergence tolerance. Default is 1e-8.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    restart : int or None, optional
        Restart parameter for GMRES. Default is 50.
    drop_tol : float, optional
        Drop tolerance for ILU factorization. Default is 0.0.
    fill_factor : float, optional
        Fill factor for ILU factorization. Default is 1.0 (ILU(0)).

    Returns
    -------
    x : numpy.ndarray
        Solution vector
    info_dict : dict
        Dictionary containing:
        - 'converged' (bool)
        - 'raw_info' (int)
        - 'niter' (int)
        - 'residual_norm' (float)
        - 'time' (float, seconds)
    """
    if method not in ("gmres", "bicgstab"):
        raise ValueError("method must be 'gmres' or 'bicgstab'")

    # Build preconditioner
    M = None
    if preconditioner.lower() == "jacobi":
        M = cpu_jacobi_precond(A)
    elif preconditioner.lower() == "ilu":
        M = cpu_ilu_precond(A, drop_tol=drop_tol, fill_factor=fill_factor)

    residuals = []

    def callback(xk):
        # GMRES/BiCGStab don't give residual, we recompute it.
        r = b - A @ xk
        residuals.append(np.linalg.norm(r))

    t0 = time.perf_counter()
    if method == "gmres":
        x, info = spla.gmres(
            A, b, M=M, tol=tol, maxiter=maxiter, restart=restart,
            callback=callback, atol=0.0
        )
    else:  # bicgstab
        x, info = spla.bicgstab(
            A, b, M=M, rtol=tol, atol=0.0, maxiter=maxiter,
            callback=callback
        )
    t1 = time.perf_counter()

    if residuals:
        res_norm = residuals[-1]
        niter = len(residuals)
    else:
        # e.g. converged in 0 iterations
        res_norm = np.linalg.norm(b - A @ x)
        niter = 0

    info_dict = {
        "converged": info == 0,
        "raw_info": info,
        "niter": niter,
        "residual_norm": res_norm,
        "time": t1 - t0,
    }
    return x, info_dict


def gpu_solve(A: sp.spmatrix,
              b: np.ndarray,
              method: str = "gmres",
              preconditioner: str = "none",
              tol: float = 1e-8,
              maxiter: int = 1000,
              restart: int | None = 50,
              drop_tol: float = 0.0,
              fill_factor: float = 1.0):
    """
    GPU solve wrapper using CuPy GMRES (and optionally BiCGStab) with
    Jacobi/ILU preconditioners.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix (will be converted to GPU)
    b : numpy.ndarray
        Right-hand side vector (will be converted to GPU)
    method : str, optional
        Solver method: "gmres" or "bicgstab". Default is "gmres".
    preconditioner : str, optional
        Preconditioner type: "none", "jacobi", or "ilu". Default is "none".
    tol : float, optional
        Convergence tolerance. Default is 1e-8.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    restart : int or None, optional
        Restart parameter for GMRES. Default is 50.
    drop_tol : float, optional
        Drop tolerance for ILU factorization. Default is 0.0.
    fill_factor : float, optional
        Fill factor for ILU factorization. Default is 1.0 (ILU(0)).

    Returns
    -------
    x : numpy.ndarray
        Solution vector (converted back to CPU)
    info_dict : dict
        Dictionary containing:
        - 'converged' (bool)
        - 'raw_info' (int)
        - 'niter' (int)
        - 'residual_norm' (float)
        - 'time' (float, seconds)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available.")

    if method not in ("gmres", "bicgstab"):
        raise ValueError("method must be 'gmres' or 'bicgstab'")

    # Convert matrix and RHS to GPU
    A_gpu = to_gpu_csr(A)
    b_gpu = to_gpu_vec(b)

    # Build preconditioner
    M_gpu = None
    if preconditioner.lower() == "jacobi":
        M_gpu = gpu_jacobi_precond(A_gpu)
    elif preconditioner.lower() == "ilu":
        M_gpu = gpu_ilu_precond(A_gpu, drop_tol=drop_tol, fill_factor=fill_factor)

    residuals = []

    def callback(xk):
        # Recompute residual on GPU
        r = b_gpu - A_gpu @ xk
        residuals.append(cp.linalg.norm(r).item())

    t0 = time.perf_counter()
    if method == "gmres":
        x_gpu, info = cpspla.gmres(
            A_gpu, b_gpu, M=M_gpu, tol=tol,
            maxiter=maxiter, restart=restart,
            callback=callback, atol=0.0
        )
    else:
        # BiCGStab may not be present in older CuPy versions
        bicgstab = getattr(cpspla, "bicgstab", None)
        if bicgstab is None:
            raise RuntimeError("cupyx.scipy.sparse.linalg.bicgstab is not available in this CuPy version.")
        x_gpu, info = bicgstab(
            A_gpu, b_gpu, M=M_gpu, tol=tol,
            maxiter=maxiter, callback=callback, atol=0.0
        )
    t1 = time.perf_counter()

    if residuals:
        res_norm = residuals[-1]
        niter = len(residuals)
    else:
        r = b_gpu - A_gpu @ x_gpu
        res_norm = cp.linalg.norm(r).item()
        niter = 0

    x = to_cpu_vec(x_gpu)

    info_dict = {
        "converged": info == 0,
        "raw_info": int(info),
        "niter": niter,
        "residual_norm": float(res_norm),
        "time": t1 - t0,
    }
    return x, info_dict


class KrylovSolver:
    """
    Unified interface for CPU/GPU Krylov solvers.
    
    This class provides a convenient interface for solving sparse linear systems
    using iterative Krylov methods (GMRES, BiCGStab) with optional preconditioning
    (Jacobi, ILU) on either CPU or GPU backends.
    """
    
    def __init__(self,
                 method: str = "gmres",
                 backend: str = "cpu",
                 preconditioner: str = "none",
                 tol: float = 1e-8,
                 maxiter: int = 1000,
                 restart: int | None = 50,
                 drop_tol: float = 0.0,
                 fill_factor: float = 1.0):
        """
        Initialize the Krylov solver.

        Parameters
        ----------
        method : str, optional
            Solver method: "gmres" or "bicgstab". Default is "gmres".
        backend : str, optional
            Backend to use: "cpu" or "gpu". Default is "cpu".
        preconditioner : str, optional
            Preconditioner type: "none", "jacobi", or "ilu". Default is "none".
        tol : float, optional
            Convergence tolerance. Default is 1e-8.
        maxiter : int, optional
            Maximum number of iterations. Default is 1000.
        restart : int or None, optional
            Restart parameter for GMRES. Default is 50.
        drop_tol : float, optional
            Drop tolerance for ILU factorization. Default is 0.0.
        fill_factor : float, optional
            Fill factor for ILU factorization. Default is 1.0 (ILU(0)).
        """
        method = method.lower()
        backend = backend.lower()
        preconditioner = preconditioner.lower()

        if method not in ("gmres", "bicgstab"):
            raise ValueError("method must be 'gmres' or 'bicgstab'")
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be 'cpu' or 'gpu'")
        if preconditioner not in ("none", "jacobi", "ilu"):
            raise ValueError("preconditioner must be 'none', 'jacobi', 'ilu'")

        if backend == "gpu" and not CUPY_AVAILABLE:
            raise RuntimeError("GPU backend requested but CuPy is not available.")

        self.method = method
        self.backend = backend
        self.preconditioner = preconditioner
        self.tol = tol
        self.maxiter = maxiter
        self.restart = restart
        self.drop_tol = drop_tol
        self.fill_factor = fill_factor

    def solve(self, A: sp.spmatrix, b: np.ndarray):
        """
        Solve Ax = b using the configured backend and method.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse matrix
        b : numpy.ndarray
            Right-hand side vector

        Returns
        -------
        x : numpy.ndarray
            Solution vector
        info_dict : dict
            Dictionary containing convergence information
        """
        if self.backend == "cpu":
            return cpu_solve(
                A, b,
                method=self.method,
                preconditioner=self.preconditioner,
                tol=self.tol,
                maxiter=self.maxiter,
                restart=self.restart,
                drop_tol=self.drop_tol,
                fill_factor=self.fill_factor,
            )
        else:
            return gpu_solve(
                A, b,
                method=self.method,
                preconditioner=self.preconditioner,
                tol=self.tol,
                maxiter=self.maxiter,
                restart=self.restart,
                drop_tol=self.drop_tol,
                fill_factor=self.fill_factor,
            )


def solve(A: sp.spmatrix,
          b: np.ndarray,
          backend: str = "gpu",
          method: str = "gmres",
          preconditioner: str = "ilu",
          tol: float = 1e-8,
          maxiter: int = 1000,
          restart: int | None = 50,
          drop_tol: float = 0.0,
          fill_factor: float = 1.0):
    """
    High-level solve function for sparse linear systems.
    
    This is the main entry point for solving Ax = b using iterative Krylov
    methods with optional preconditioning on CPU or GPU backends.
    
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse matrix
    b : numpy.ndarray
        Right-hand side vector
    backend : str, optional
        Backend to use: "cpu" or "gpu". Default is "gpu".
    method : str, optional
        Solver method: "gmres" or "bicgstab". Default is "gmres".
    preconditioner : str, optional
        Preconditioner type: "none", "jacobi", or "ilu". Default is "ilu".
    tol : float, optional
        Convergence tolerance. Default is 1e-8.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    restart : int or None, optional
        Restart parameter for GMRES. Default is 50.
    drop_tol : float, optional
        Drop tolerance for ILU factorization. Default is 0.0.
    fill_factor : float, optional
        Fill factor for ILU factorization. Default is 1.0 (ILU(0)).

    Returns
    -------
    x : numpy.ndarray
        Solution vector
    info_dict : dict
        Dictionary containing:
        - 'converged' (bool): Whether the solver converged
        - 'raw_info' (int): Raw convergence info from the solver
        - 'niter' (int): Number of iterations
        - 'residual_norm' (float): Final residual norm
        - 'time' (float): Time taken in seconds
    
    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from krylov_solver import solve
    >>> 
    >>> # Create a test matrix
    >>> A = sp.random(1000, 1000, density=0.01, format='csr')
    >>> A = A + sp.eye(1000) * 1.1  # Make it diagonally dominant
    >>> b = np.random.rand(1000)
    >>> 
    >>> # Solve on GPU with BiCGStab and ILU preconditioning
    >>> x, info = solve(A, b, backend="gpu", method="bicgstab", preconditioner="ilu")
    >>> print(f"Converged: {info['converged']}, Iterations: {info['niter']}")
    """
    solver = KrylovSolver(
        method=method,
        backend=backend,
        preconditioner=preconditioner,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        drop_tol=drop_tol,
        fill_factor=fill_factor,
    )
    return solver.solve(A, b)


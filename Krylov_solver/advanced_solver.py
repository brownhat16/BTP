"""
Advanced solver with backend abstraction, async support, and enhanced features.
"""

import threading
import queue
from typing import Optional, Callable, Any, Union
import numpy as np
import scipy.sparse as sp

from .backends.base import Backend, BackendRegistry
from .backends.scipy_backend import SciPyBackend
try:
    from .backends.cupy_backend import CuPyBackend
except ImportError:
    CuPyBackend = None
from .convergence import ConvergenceInfo
from .utils import poisson_2d, load_matrix_market

# Register available backends
BackendRegistry.register("scipy", SciPyBackend)
if CuPyBackend is not None:
    BackendRegistry.register("cupy", CuPyBackend)

try:
    from .backends import PETScBackend
    BackendRegistry.register("petsc", PETScBackend)
except:
    pass


class AdvancedLinearSolver:
    """
    Advanced linear solver with backend abstraction and enhanced features.
    
    Supports:
    - Multiple backends (SciPy, CuPy, PETSc, Hypre)
    - Preconditioner caching and reuse
    - Callback functions for monitoring
    - Async solving
    - Multiple input types (NumPy, CuPy, PyTorch, JAX)
    - Device selection for multi-GPU
    """
    
    def __init__(self,
                 method: str = "gmres",
                 backend: Optional[str] = None,
                 preconditioner: Union[str, Any] = "ilu",
                 tol: float = 1e-8,
                 rtol: Optional[float] = None,
                 atol: float = 0.0,
                 maxiter: int = 1000,
                 restart: Optional[int] = None,
                 device_id: Optional[int] = None,
                 **preconditioner_kwargs):
        """
        Initialize the advanced linear solver.
        
        Parameters
        ----------
        method : str
            Solver method ("gmres", "bicgstab", etc.)
        backend : str, optional
            Backend to use ("scipy", "cupy", "petsc", "hypre", or None for auto-select)
        preconditioner : str, LinearOperator, or callable
            Preconditioner specification
        tol : float
            Tolerance (used as rtol if rtol is None)
        rtol : float, optional
            Relative tolerance
        atol : float
            Absolute tolerance
        maxiter : int
            Maximum iterations
        restart : int, optional
            GMRES restart parameter (default: 50 for GMRES)
        device_id : int, optional
            GPU device ID (for GPU backends)
        **preconditioner_kwargs
            Additional preconditioner parameters (e.g., drop_tol, fill_factor)
        """
        self.method = method.lower()
        self.preconditioner = preconditioner
        self.tol = tol
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.restart = restart
        self.preconditioner_kwargs = preconditioner_kwargs
        
        # Select backend
        if backend is None:
            prefer_gpu = (device_id is not None)
            backend = BackendRegistry.auto_select(prefer_gpu=prefer_gpu)
        
        self.backend_name = backend
        self.backend: Backend = BackendRegistry.get_backend(backend, device_id=device_id)
        
        # Verify method and preconditioner are supported
        if not self.backend.supports_method(self.method):
            # Provide helpful error message with alternatives
            available_methods = []
            for method in ["gmres", "bicgstab"]:
                if self.backend.supports_method(method):
                    available_methods.append(method)
            if available_methods:
                raise ValueError(
                    f"Method {self.method} not supported by {backend} backend. "
                    f"Available methods: {available_methods}"
                )
            else:
                raise ValueError(f"Method {self.method} not supported by {backend} backend")
        
        if isinstance(preconditioner, str):
            if not self.backend.supports_preconditioner(preconditioner):
                available_precs = []
                for prec in ["none", "jacobi", "ilu"]:
                    if self.backend.supports_preconditioner(prec):
                        available_precs.append(prec)
                if available_precs:
                    raise ValueError(
                        f"Preconditioner {preconditioner} not supported by {backend} backend. "
                        f"Available preconditioners: {available_precs}"
                    )
                else:
                    raise ValueError(f"Preconditioner {preconditioner} not supported by {backend} backend")
    
    def solve(self,
              A: Any,
              b: Any,
              callback: Optional[Callable] = None,
              reuse_preconditioner: bool = True) -> tuple[Any, ConvergenceInfo]:
        """
        Solve Ax = b.
        
        Parameters
        ----------
        A : sparse matrix
            System matrix (SciPy, CuPy, or compatible)
        b : array-like
            Right-hand side vector (NumPy, CuPy, PyTorch, JAX, or compatible)
        callback : callable, optional
            Callback function(xk) called each iteration.
            Note: May impact performance if called frequently.
        reuse_preconditioner : bool
            Whether to reuse cached preconditioner if available
        
        Returns
        -------
        x : array
            Solution vector (same type as input b)
        info : ConvergenceInfo
            Convergence information
        """
        # Convert input types if needed
        A, b = self._prepare_inputs(A, b)
        
        # For GPU backends, preconditioner creation happens inside solve()
        # to ensure matrix is converted to GPU format first
        # For CPU backends, we can create it here
        if self.backend_name == "cupy":
            # Pass preconditioner spec to backend, let it handle conversion
            M = self.preconditioner
        else:
            # Get or create preconditioner for CPU backends
            if reuse_preconditioner:
                M = self.backend.create_preconditioner(
                    A, self.preconditioner, **self.preconditioner_kwargs
                )
            else:
                M = self.backend._create_preconditioner_impl(
                    A, self.preconditioner, **self.preconditioner_kwargs
                )
        
        # Solve
        x, info = self.backend.solve(
            A, b,
            method=self.method,
            preconditioner=M,
            tol=self.tol,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            restart=self.restart,
            callback=callback,
            **self.preconditioner_kwargs  # Pass preconditioner kwargs to backend
        )
        
        return x, info
    
    def solve_async(self,
                    A: Any,
                    b: Any,
                    callback: Optional[Callable] = None) -> 'AsyncSolveHandle':
        """
        Solve Ax = b asynchronously.
        
        Parameters
        ----------
        A : sparse matrix
            System matrix
        b : array-like
            Right-hand side vector
        callback : callable, optional
            Callback function for monitoring
        
        Returns
        -------
        handle : AsyncSolveHandle
            Handle to query status and get results
        """
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def solve_thread():
            try:
                x, info = self.solve(A, b, callback=callback)
                result_queue.put((x, info))
            except Exception as e:
                error_queue.put(e)
        
        thread = threading.Thread(target=solve_thread, daemon=True)
        thread.start()
        
        return AsyncSolveHandle(thread, result_queue, error_queue)
    
    def factorize(self, A: Any):
        """
        Pre-factorize the matrix (e.g., compute ILU) and cache it.
        
        This is useful when solving multiple systems with the same matrix.
        
        Parameters
        ----------
        A : sparse matrix
            System matrix to factorize
        """
        A, _ = self._prepare_inputs(A, np.zeros(A.shape[0]))
        
        # For GPU backends, convert to GPU format first
        if self.backend_name == "cupy":
            if hasattr(self.backend, '_to_gpu_csr'):
                A = self.backend._to_gpu_csr(A)
        
        self.backend.create_preconditioner(
            A, self.preconditioner, **self.preconditioner_kwargs
        )
    
    def clear_cache(self):
        """Clear the preconditioner cache."""
        self.backend.clear_cache()
    
    def _prepare_inputs(self, A: Any, b: Any) -> tuple[Any, Any]:
        """
        Prepare and convert inputs to appropriate types.
        
        Handles NumPy, CuPy, PyTorch, JAX, and SciPy types.
        """
        # Handle PyTorch
        try:
            import torch
            if isinstance(A, torch.Tensor):
                A = A.cpu().numpy()
            if isinstance(b, torch.Tensor):
                b = b.cpu().numpy()
        except ImportError:
            pass
        
        # Handle JAX
        try:
            import jax.numpy as jnp
            if isinstance(A, jnp.ndarray):
                A = np.asarray(A)
            if isinstance(b, jnp.ndarray):
                b = np.asarray(b)
        except ImportError:
            pass
        
        # Ensure A is sparse matrix
        if not sp.issparse(A):
            try:
                import cupy as cp
                if isinstance(A, cp.ndarray):
                    # Dense CuPy array - convert to sparse if needed
                    # For now, assume it's already in the right format
                    pass
                else:
                    # Dense NumPy array - convert to sparse
                    A = sp.csr_matrix(A)
            except ImportError:
                if isinstance(A, np.ndarray):
                    A = sp.csr_matrix(A)
        
        return A, b


class AsyncSolveHandle:
    """Handle for asynchronous solve operations."""
    
    def __init__(self, thread: threading.Thread, result_queue: queue.Queue, error_queue: queue.Queue):
        self.thread = thread
        self.result_queue = result_queue
        self.error_queue = error_queue
        self._result = None
        self._error = None
    
    def is_done(self) -> bool:
        """Check if solve is complete."""
        if self._result is not None or self._error is not None:
            return True
        if not self.error_queue.empty():
            self._error = self.error_queue.get()
            return True
        if not self.result_queue.empty():
            self._result = self.result_queue.get()
            return True
        return not self.thread.is_alive()
    
    def wait(self, timeout: Optional[float] = None) -> tuple[Any, ConvergenceInfo]:
        """
        Wait for solve to complete and return result.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds
        
        Returns
        -------
        x : array
            Solution vector
        info : ConvergenceInfo
            Convergence information
        
        Raises
        ------
        TimeoutError
            If timeout is exceeded
        RuntimeError
            If solve failed
        """
        self.thread.join(timeout=timeout)
        
        if self._error:
            raise RuntimeError(f"Solve failed: {self._error}") from self._error
        
        if not self.result_queue.empty():
            self._result = self.result_queue.get()
        
        if self._result is None:
            if self.thread.is_alive():
                raise TimeoutError("Solve did not complete within timeout")
            else:
                raise RuntimeError("Solve thread terminated without result")
        
        return self._result
    
    def get_result(self) -> Optional[tuple[Any, ConvergenceInfo]]:
        """
        Get result if available, otherwise return None.
        
        Returns
        -------
        result : tuple or None
            (x, info) if available, None otherwise
        """
        if self._result:
            return self._result
        if not self.result_queue.empty():
            self._result = self.result_queue.get()
            return self._result
        return None


# Convenience function for backward compatibility
def solve(A: sp.spmatrix,
          b: np.ndarray,
          backend: str = "auto",
          method: str = "gmres",
          preconditioner: Union[str, Any] = "ilu",
          tol: float = 1e-8,
          rtol: Optional[float] = None,
          atol: float = 0.0,
          maxiter: int = 1000,
          restart: Optional[int] = None,
          device_id: Optional[int] = None,
          callback: Optional[Callable] = None,
          **kwargs) -> tuple[Any, ConvergenceInfo]:
    """
    High-level solve function with enhanced features.
    
    This is an enhanced version of the original solve() function with:
    - Backend abstraction
    - Better convergence info
    - Callback support
    - Multiple input types
    
    Parameters
    ----------
    A : sparse matrix
        System matrix
    b : array-like
        Right-hand side vector
    backend : str
        Backend to use ("auto", "scipy", "cupy", "petsc", "hypre")
    method : str
        Solver method ("gmres", "bicgstab")
    preconditioner : str, LinearOperator, or callable
        Preconditioner specification
    tol : float
        Tolerance
    rtol : float, optional
        Relative tolerance (overrides tol if provided)
    atol : float
        Absolute tolerance
    maxiter : int
        Maximum iterations
    restart : int, optional
        GMRES restart parameter
    device_id : int, optional
        GPU device ID
    callback : callable, optional
        Callback function(xk) for monitoring
    **kwargs
        Additional preconditioner parameters
    
    Returns
    -------
    x : array
        Solution vector
    info : ConvergenceInfo
        Convergence information
    """
    if backend == "auto":
        backend = None
    
    solver = AdvancedLinearSolver(
        method=method,
        backend=backend,
        preconditioner=preconditioner,
        tol=tol,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        restart=restart,
        device_id=device_id,
        **kwargs
    )
    
    return solver.solve(A, b, callback=callback)


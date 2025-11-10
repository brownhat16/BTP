"""
Base backend interface for solver implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict
import numpy as np
import scipy.sparse as sp

from ..convergence import ConvergenceInfo


class Backend(ABC):
    """
    Abstract base class for solver backends.
    
    Each backend implements the solver using a specific library
    (e.g., SciPy, CuPy, PETSc, Hypre).
    """
    
    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize the backend.
        
        Parameters
        ----------
        device_id : int, optional
            GPU device ID (for GPU backends)
        """
        self.device_id = device_id
        self._preconditioner_cache: Dict[str, Any] = {}
    
    @abstractmethod
    def solve(self,
              A: Any,
              b: Any,
              method: str = "gmres",
              preconditioner: Any = None,
              tol: float = 1e-8,
              rtol: Optional[float] = None,
              atol: float = 0.0,
              maxiter: int = 1000,
              restart: Optional[int] = None,
              callback: Optional[Callable] = None,
              **kwargs) -> tuple[Any, ConvergenceInfo]:
        """
        Solve Ax = b.
        
        Parameters
        ----------
        A : sparse matrix
            System matrix
        b : array-like
            Right-hand side vector
        method : str
            Solver method ("gmres", "bicgstab", etc.)
        preconditioner : Any
            Preconditioner (string, LinearOperator, or callable)
        tol : float
            Tolerance (used as rtol if rtol is None)
        rtol : float, optional
            Relative tolerance
        atol : float
            Absolute tolerance
        maxiter : int
            Maximum iterations
        restart : int, optional
            GMRES restart parameter
        callback : callable, optional
            Callback function called each iteration
        **kwargs
            Additional backend-specific parameters
        
        Returns
        -------
        x : array
            Solution vector (same type as input)
        info : ConvergenceInfo
            Convergence information
        """
        pass
    
    @abstractmethod
    def supports_method(self, method: str) -> bool:
        """Check if backend supports a given solver method."""
        pass
    
    @abstractmethod
    def supports_preconditioner(self, preconditioner: str) -> bool:
        """Check if backend supports a given preconditioner type."""
        pass
    
    def create_preconditioner(self,
                             A: Any,
                             preconditioner: Any,
                             **kwargs) -> Any:
        """
        Create or retrieve a cached preconditioner.
        
        Parameters
        ----------
        A : sparse matrix
            System matrix
        preconditioner : Any
            Preconditioner specification (string, LinearOperator, or callable)
        **kwargs
            Preconditioner-specific parameters
        
        Returns
        -------
        M : preconditioner object
            Preconditioner that can be applied
        """
        # Check cache
        cache_key = self._get_preconditioner_key(A, preconditioner, **kwargs)
        if cache_key in self._preconditioner_cache:
            return self._preconditioner_cache[cache_key]
        
        # Create new preconditioner
        M = self._create_preconditioner_impl(A, preconditioner, **kwargs)
        
        # Cache it
        self._preconditioner_cache[cache_key] = M
        return M
    
    def _get_preconditioner_key(self, A: Any, preconditioner: Any, **kwargs) -> str:
        """Generate a cache key for the preconditioner."""
        import hashlib
        # Use matrix shape and nnz as part of key
        key_parts = [
            str(A.shape),
            str(getattr(A, 'nnz', None)),
            str(preconditioner),
            str(sorted(kwargs.items()))
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @abstractmethod
    def _create_preconditioner_impl(self,
                                    A: Any,
                                    preconditioner: Any,
                                    **kwargs) -> Any:
        """Implementation-specific preconditioner creation."""
        pass
    
    def clear_cache(self):
        """Clear the preconditioner cache."""
        self._preconditioner_cache.clear()


class BackendRegistry:
    """Registry for available backends."""
    
    _backends: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: type):
        """Register a backend class."""
        cls._backends[name] = backend_class
    
    @classmethod
    def get_backend(cls, name: str, **kwargs) -> Backend:
        """Get an instance of a backend."""
        if name not in cls._backends:
            raise ValueError(f"Unknown backend: {name}. Available: {list(cls._backends.keys())}")
        return cls._backends[name](**kwargs)
    
    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backends."""
        return list(cls._backends.keys())
    
    @classmethod
    def auto_select(cls, prefer_gpu: bool = True) -> str:
        """
        Automatically select the best available backend.
        
        Parameters
        ----------
        prefer_gpu : bool
            Prefer GPU backends if available
        
        Returns
        -------
        backend_name : str
            Name of the selected backend
        """
        if prefer_gpu:
            # Try GPU backends first
            for backend in ["cupy", "petsc", "hypre"]:
                if backend in cls._backends:
                    try:
                        # Try to instantiate to check availability
                        cls.get_backend(backend)
                        return backend
                    except:
                        continue
        
        # Fall back to CPU backends
        for backend in ["scipy", "petsc"]:
            if backend in cls._backends:
                try:
                    cls.get_backend(backend)
                    return backend
                except:
                    continue
        
        raise RuntimeError("No available backends found")


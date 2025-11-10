"""
SciPy backend implementation.
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Callable, Any

from .base import Backend
from ..convergence import ConvergenceInfo
from ..preconditioners import cpu_jacobi_precond, cpu_ilu_precond


class SciPyBackend(Backend):
    """SciPy-based CPU backend."""
    
    def __init__(self, device_id: Optional[int] = None):
        super().__init__(device_id)
        if device_id is not None:
            raise ValueError("SciPy backend does not support device_id (CPU only)")
    
    def supports_method(self, method: str) -> bool:
        """Check if backend supports a given solver method."""
        return method.lower() in ("gmres", "bicgstab")
    
    def supports_preconditioner(self, preconditioner: str) -> bool:
        """Check if backend supports a given preconditioner type."""
        return preconditioner.lower() in ("none", "jacobi", "ilu")
    
    def solve(self,
              A: sp.spmatrix,
              b: np.ndarray,
              method: str = "gmres",
              preconditioner: Any = None,
              tol: float = 1e-8,
              rtol: Optional[float] = None,
              atol: float = 0.0,
              maxiter: int = 1000,
              restart: Optional[int] = None,
              callback: Optional[Callable] = None,
              **kwargs) -> tuple[np.ndarray, ConvergenceInfo]:
        """
        Solve using SciPy.
        
        Parameters
        ----------
        callback : callable, optional
            Callback function(xk) called each iteration.
            Note: This may impact performance if called frequently.
        """
        method = method.lower()
        if not self.supports_method(method):
            raise ValueError(f"Method {method} not supported by SciPy backend")
        
        # Use rtol if provided, otherwise use tol
        if rtol is None:
            rtol = tol
        
        # Build preconditioner
        M = None
        setup_time = 0.0
        
        if preconditioner is not None:
            if isinstance(preconditioner, str):
                t0_setup = time.perf_counter()
                M = self.create_preconditioner(A, preconditioner, **kwargs)
                setup_time = time.perf_counter() - t0_setup
            elif hasattr(preconditioner, 'matvec'):
                # LinearOperator
                M = preconditioner
            elif callable(preconditioner):
                # User-defined function
                from scipy.sparse.linalg import LinearOperator
                n = A.shape[0]
                M = LinearOperator((n, n), matvec=preconditioner)
            else:
                M = preconditioner
        
        # Track residuals for callback
        residuals = []
        b_norm = np.linalg.norm(b)
        
        def wrapped_callback(xk):
            # Ensure xk is the right shape
            try:
                # Convert to array if needed
                if not isinstance(xk, np.ndarray):
                    xk = np.asarray(xk)
                # Ensure it's 1D
                if xk.ndim == 0:
                    # Scalar - this shouldn't happen, but handle it
                    return
                if xk.ndim > 1:
                    xk = xk.flatten()
                
                if callback is not None:
                    callback(xk)
                
                # Track residual - ensure shapes match
                if xk.shape[0] == b.shape[0]:
                    r = b - A @ xk
                    res_norm = np.linalg.norm(r)
                    residuals.append(res_norm)
            except Exception as e:
                # If callback fails, just skip this iteration's tracking
                # Don't break the solver
                pass
        
        # Always use callback to track residuals, even if user didn't provide one
        tracking_callback = wrapped_callback
        
        # Solve
        t0 = time.perf_counter()
        if method == "gmres":
            if restart is None:
                restart = 50
            x, info = spla.gmres(
                A, b, M=M, rtol=rtol, atol=atol,
                maxiter=maxiter, restart=restart,
                callback=tracking_callback
            )
        else:  # bicgstab
            x, info = spla.bicgstab(
                A, b, M=M, rtol=rtol, atol=atol,
                maxiter=maxiter,
                callback=tracking_callback
            )
        solve_time = time.perf_counter() - t0
        
        # Compute final residual and iteration count
        # SciPy's info: 0 = converged, >0 = number of iterations when stopped, <0 = error
        if residuals:
            res_norm = residuals[-1]
            niter = len(residuals)
        else:
            # Callback wasn't called (rare, but can happen)
            r = b - A @ x
            res_norm = np.linalg.norm(r)
            # Use info to determine iterations
            if info == 0:
                # Converged - try to estimate from residual
                # If residual is very small, likely converged quickly
                if res_norm < rtol * b_norm or res_norm < atol:
                    niter = 1  # At least 1 iteration
                else:
                    niter = maxiter  # Unknown, use maxiter as fallback
            elif info > 0:
                niter = info  # Number of iterations performed
            else:
                niter = abs(info)  # Error code, use absolute value
        
        # Determine convergence reason
        if info == 0:
            reason = "Converged"
        elif info > 0:
            reason = f"Did not converge in {info} iterations"
        else:
            reason = "Breakdown or error"
        
        conv_info = ConvergenceInfo(
            converged=(info == 0),
            iterations=niter,
            residual_norm=res_norm,
            relative_residual=res_norm / b_norm if b_norm > 0 else res_norm,
            solve_time=solve_time,
            setup_time=setup_time,
            reason=reason,
            raw_info=int(info)
        )
        
        return x, conv_info
    
    def _create_preconditioner_impl(self,
                                    A: sp.spmatrix,
                                    preconditioner: Any,
                                    **kwargs) -> Any:
        """Create a preconditioner using SciPy."""
        # Handle string-based preconditioners
        if isinstance(preconditioner, str):
            prec_type = preconditioner.lower()
            if prec_type == "jacobi":
                return cpu_jacobi_precond(A)
            elif prec_type == "ilu":
                drop_tol = kwargs.get("drop_tol", 0.0)
                fill_factor = kwargs.get("fill_factor", 1.0)
                return cpu_ilu_precond(A, drop_tol=drop_tol, fill_factor=fill_factor)
            elif prec_type == "none":
                return None
            else:
                raise ValueError(f"Unknown preconditioner type: {preconditioner}")
        
        # For non-string preconditioners (LinearOperator, callable, etc.),
        # return as-is - they'll be handled in solve()
        return preconditioner


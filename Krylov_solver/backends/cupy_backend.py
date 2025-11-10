"""
CuPy backend implementation for GPU solving.
"""

import time
import numpy as np
import scipy.sparse as sp
from typing import Optional, Callable, Any

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpspla
    from cupyx.scipy.sparse.linalg import LinearOperator as CuLinearOperator
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .base import Backend
from ..convergence import ConvergenceInfo
from ..preconditioners import gpu_jacobi_precond, gpu_ilu_precond


class CuPyBackend(Backend):
    """CuPy-based GPU backend."""
    
    def __init__(self, device_id: Optional[int] = None):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available. Install with: pip install cupy")
        super().__init__(device_id)
        
        # Set device if specified
        if device_id is not None:
            with cp.cuda.Device(device_id):
                pass  # Just verify device is available
    
    def supports_method(self, method: str) -> bool:
        """Check if backend supports a given solver method."""
        method = method.lower()
        if method == "gmres":
            return True
        elif method == "bicgstab":
            # Check if bicgstab is available and callable
            try:
                bicgstab = getattr(cpspla, "bicgstab", None)
                if bicgstab is None:
                    return False
                # Check if it's actually callable (some versions have it but it's not implemented)
                return callable(bicgstab)
            except:
                return False
        return False
    
    def supports_preconditioner(self, preconditioner: str) -> bool:
        """Check if backend supports a given preconditioner type."""
        return preconditioner.lower() in ("none", "jacobi", "ilu")
    
    def _to_gpu(self, obj: Any) -> Any:
        """Convert object to GPU."""
        if isinstance(obj, cp.ndarray) or hasattr(obj, 'get'):
            return obj
        return cp.asarray(obj)
    
    def _to_gpu_csr(self, A: sp.spmatrix) -> Any:
        """Convert SciPy sparse matrix to CuPy CSR."""
        A_csr = A.tocsr()
        data = cp.asarray(A_csr.data)
        indices = cp.asarray(A_csr.indices)
        indptr = cp.asarray(A_csr.indptr)
        return cpsp.csr_matrix((data, indices, indptr), shape=A_csr.shape)
    
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
        Solve using CuPy on GPU.
        
        Parameters
        ----------
        callback : callable, optional
            Callback function(xk) called each iteration.
            Note: This requires transferring data to CPU, which may impact performance.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available")
        
        method = method.lower()
        if not self.supports_method(method):
            available = []
            for m in ["gmres", "bicgstab"]:
                if self.supports_method(m):
                    available.append(m)
            raise ValueError(
                f"Method {method} not supported by CuPy backend. "
                f"Available methods: {available}"
            )
        
        # Set device context
        device_context = cp.cuda.Device(self.device_id) if self.device_id is not None else None
        if device_context:
            device_context.use()
        
        # Convert to GPU
        if not isinstance(A, (cpsp.csr_matrix, cpsp.csc_matrix)):
            A_gpu = self._to_gpu_csr(A)
        else:
            A_gpu = A
        
        b_gpu = self._to_gpu(b)
        b_norm = cp.linalg.norm(b_gpu).item()
        
        # Use rtol if provided, otherwise use tol
        if rtol is None:
            rtol = tol
        
        # Build preconditioner (must use A_gpu, not original A)
        M_gpu = None
        setup_time = 0.0
        
        if preconditioner is not None:
            if isinstance(preconditioner, str):
                t0_setup = time.perf_counter()
                # Create preconditioner using GPU matrix
                M_gpu = self.create_preconditioner(A_gpu, preconditioner, **kwargs)
                setup_time = time.perf_counter() - t0_setup
            elif hasattr(preconditioner, 'matvec'):
                # LinearOperator (should be CuPy LinearOperator for GPU)
                M_gpu = preconditioner
            elif callable(preconditioner):
                # User-defined function - wrap as CuPy LinearOperator
                n = A_gpu.shape[0]
                def gpu_matvec(x):
                    return self._to_gpu(preconditioner(cp.asnumpy(x)))
                M_gpu = CuLinearOperator((n, n), matvec=gpu_matvec)
            else:
                M_gpu = preconditioner
        
        # Track residuals
        residuals = []
        
        def wrapped_callback(xk):
            # Ensure xk is the right shape (CuPy callbacks might pass different formats)
            try:
                # Convert to array if needed
                if not isinstance(xk, cp.ndarray):
                    xk = cp.asarray(xk)
                # Ensure it's 1D
                if xk.ndim == 0:
                    # Scalar - this shouldn't happen, but handle it
                    return
                if xk.ndim > 1:
                    xk = xk.flatten()
                
                if callback is not None:
                    # Transfer to CPU for callback (may be slow)
                    xk_cpu = cp.asnumpy(xk)
                    callback(xk_cpu)
                
                # Track residual - ensure shapes match
                if xk.shape[0] == b_gpu.shape[0]:
                    r = b_gpu - A_gpu @ xk
                    res_norm = cp.linalg.norm(r).item()
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
            # CuPy GMRES uses 'tol' instead of 'rtol'/'atol'
            # Combine rtol and atol into a single tolerance
            # Use rtol if provided, otherwise use tol
            combined_tol = rtol if rtol is not None else tol
            # If atol is significant, we might need to adjust, but typically rtol is used
            x_gpu, info = cpspla.gmres(
                A_gpu, b_gpu, M=M_gpu, tol=combined_tol,
                maxiter=maxiter, restart=restart,
                callback=tracking_callback
            )
        elif method == "bicgstab":
            bicgstab = getattr(cpspla, "bicgstab", None)
            if bicgstab is None:
                raise RuntimeError(
                    "cupyx.scipy.sparse.linalg.bicgstab is not available in this CuPy version. "
                    "Try using method='gmres' instead, or upgrade CuPy."
                )
            # Check if it's actually callable (some CuPy versions have it but it's not implemented)
            if not callable(bicgstab):
                raise RuntimeError(
                    "cupyx.scipy.sparse.linalg.bicgstab exists but is not callable. "
                    "Try using method='gmres' instead."
                )
            # CuPy BiCGStab also uses 'tol' instead of 'rtol'/'atol'
            combined_tol = rtol if rtol is not None else tol
            x_gpu, info = bicgstab(
                A_gpu, b_gpu, M=M_gpu, tol=combined_tol,
                maxiter=maxiter,
                callback=tracking_callback
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        solve_time = time.perf_counter() - t0
        
        # Compute final residual and iteration count
        # CuPy's info: 0 = converged, >0 = number of iterations when stopped, <0 = error
        if residuals:
            res_norm = residuals[-1]
            niter = len(residuals)
        else:
            # Callback wasn't called (rare, but can happen)
            r = b_gpu - A_gpu @ x_gpu
            res_norm = cp.linalg.norm(r).item()
            # Use info to determine iterations
            if info == 0:
                # Converged - try to estimate from residual
                # If residual is very small, likely converged quickly
                if res_norm < combined_tol * b_norm:
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
        
        # Return same type as input
        if isinstance(b, cp.ndarray):
            x = x_gpu  # Keep on GPU
        else:
            x = cp.asnumpy(x_gpu)  # Transfer to CPU
        
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
                                    A: Any,
                                    preconditioner: Any,
                                    **kwargs) -> Any:
        """Create a preconditioner using CuPy."""
        # Handle string-based preconditioners
        if isinstance(preconditioner, str):
            prec_type = preconditioner.lower()
            if prec_type == "jacobi":
                return gpu_jacobi_precond(A)
            elif prec_type == "ilu":
                drop_tol = kwargs.get("drop_tol", 0.0)
                fill_factor = kwargs.get("fill_factor", 1.0)
                return gpu_ilu_precond(A, drop_tol=drop_tol, fill_factor=fill_factor)
            elif prec_type == "none":
                return None
            else:
                raise ValueError(f"Unknown preconditioner type: {preconditioner}")
        
        # For non-string preconditioners, return as-is
        return preconditioner

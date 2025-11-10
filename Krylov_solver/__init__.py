"""
GPU-Accelerated Krylov Solvers for Large Sparse Linear Systems

This package provides iterative Krylov methods (GMRES, BiCGStab) with
preconditioning (Jacobi, ILU) for solving large sparse linear systems,
with support for both CPU (SciPy) and GPU (CuPy) backends.

Enhanced features:
- Backend abstraction (SciPy, CuPy, PETSc, Hypre)
- Preconditioner caching and reuse
- Callback support for monitoring
- Async solving
- Multiple input types (NumPy, CuPy, PyTorch, JAX)
- Device selection for multi-GPU
"""

# Backward compatibility - original API
from .solver import solve as solve_legacy, KrylovSolver
from .utils import load_matrix_market, poisson_2d

# Enhanced API
from .advanced_solver import solve, AdvancedLinearSolver, AsyncSolveHandle
from .convergence import ConvergenceInfo
from .backends import BackendRegistry

__version__ = "0.2.0"
__all__ = [
    # Original API (backward compatible)
    "solve_legacy",
    "KrylovSolver",
    # Enhanced API
    "solve",
    "AdvancedLinearSolver",
    "AsyncSolveHandle",
    "ConvergenceInfo",
    "BackendRegistry",
    # Utilities
    "load_matrix_market",
    "poisson_2d",
]


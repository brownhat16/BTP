"""
Backend implementations for different solver libraries.
"""

from .base import Backend, BackendRegistry
from .cupy_backend import CuPyBackend
from .scipy_backend import SciPyBackend

__all__ = ["Backend", "BackendRegistry", "CuPyBackend", "SciPyBackend"]

# Try to import optional backends
try:
    from .petsc_backend import PETScBackend
    __all__.append("PETScBackend")
except ImportError:
    pass

try:
    from .hypre_backend import HypreBackend
    __all__.append("HypreBackend")
except ImportError:
    pass


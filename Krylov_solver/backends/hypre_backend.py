"""
Hypre backend implementation (optional, requires hypre-py or similar).
"""

# Placeholder for Hypre backend
# Hypre is typically accessed via PETSc or through its own Python bindings

from typing import Optional, Callable, Any
from .base import Backend
from ..convergence import ConvergenceInfo


class HypreBackend(Backend):
    """
    Hypre backend for high-performance AMG and other solvers.
    
    Note: Hypre is typically used through PETSc, but can also be
    accessed directly if bindings are available.
    """
    
    def __init__(self, device_id: Optional[int] = None):
        super().__init__(device_id)
        # Check for Hypre availability
        # This would require hypre-py or similar bindings
        raise NotImplementedError(
            "Hypre backend is not implemented. "
            "Hypre is typically accessed through PETSc."
        )
    
    def supports_method(self, method: str) -> bool:
        """Check if backend supports a given solver method."""
        return False  # Not implemented
    
    def supports_preconditioner(self, preconditioner: str) -> bool:
        """Check if backend supports a given preconditioner type."""
        return False  # Not implemented
    
    def solve(self, *args, **kwargs):
        """Placeholder."""
        raise NotImplementedError("Hypre backend not implemented")
    
    def _create_preconditioner_impl(self, *args, **kwargs):
        """Placeholder."""
        raise NotImplementedError("Hypre backend not implemented")


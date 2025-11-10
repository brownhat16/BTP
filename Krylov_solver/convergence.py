"""
Convergence information and result objects.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConvergenceInfo:
    """
    Information about the convergence of a linear solver.
    
    Attributes
    ----------
    converged : bool
        Whether the solver converged to the specified tolerance
    iterations : int
        Number of iterations performed
    residual_norm : float
        Final residual norm ||b - Ax||
    relative_residual : float
        Final relative residual norm ||b - Ax|| / ||b||
    solve_time : float
        Time taken for the solve (seconds)
    setup_time : float
        Time taken for preconditioner setup (seconds)
    reason : str
        Human-readable reason for termination
    raw_info : int
        Raw convergence info from the underlying solver
    """
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    solve_time: float
    setup_time: float = 0.0
    reason: str = ""
    raw_info: int = 0
    
    def __str__(self):
        status = "Converged" if self.converged else "Not converged"
        return (
            f"{status} in {self.iterations} iterations\n"
            f"  Residual norm: {self.residual_norm:.2e}\n"
            f"  Relative residual: {self.relative_residual:.2e}\n"
            f"  Solve time: {self.solve_time:.4f}s"
        )
    
    def to_dict(self):
        """Convert to dictionary for backward compatibility."""
        return {
            "converged": self.converged,
            "niter": self.iterations,
            "residual_norm": self.residual_norm,
            "relative_residual": self.relative_residual,
            "time": self.solve_time,
            "setup_time": self.setup_time,
            "reason": self.reason,
            "raw_info": self.raw_info,
        }


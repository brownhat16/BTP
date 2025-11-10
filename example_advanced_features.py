"""
Example demonstrating advanced features of the krylov_solver package.
"""

import numpy as np
import scipy.sparse as sp
from krylov_solver import (
    solve, AdvancedLinearSolver, AsyncSolveHandle,
    ConvergenceInfo, BackendRegistry, poisson_2d
)


def example_enhanced_solve():
    """Example using the enhanced solve() function."""
    print("=" * 70)
    print("Example 1: Enhanced solve() with ConvergenceInfo")
    print("=" * 70)
    
    A = poisson_2d(100, 100)
    b = np.random.rand(A.shape[0])
    
    # Solve with enhanced API
    x, info = solve(
        A, b,
        backend="scipy",
        method="bicgstab",
        preconditioner="ilu",
        drop_tol=1e-4,
        fill_factor=1.0
    )
    
    # info is now a ConvergenceInfo object with rich information
    print(f"Converged: {info.converged}")
    print(f"Iterations: {info.iterations}")
    print(f"Residual norm: {info.residual_norm:.2e}")
    print(f"Relative residual: {info.relative_residual:.2e}")
    print(f"Solve time: {info.solve_time:.4f}s")
    print(f"Setup time: {info.setup_time:.4f}s")
    print(f"Reason: {info.reason}")
    print()
    print("Full info object:")
    print(info)
    print()


def example_advanced_solver_class():
    """Example using AdvancedLinearSolver class."""
    print("=" * 70)
    print("Example 2: AdvancedLinearSolver with Preconditioner Caching")
    print("=" * 70)
    
    A = poisson_2d(80, 80)
    b1 = np.random.rand(A.shape[0])
    b2 = np.random.rand(A.shape[0])
    b3 = np.random.rand(A.shape[0])
    
    # Create solver - preconditioner will be cached
    solver = AdvancedLinearSolver(
        method="bicgstab",
        backend="scipy",
        preconditioner="ilu",
        drop_tol=1e-4
    )
    
    # First solve - preconditioner is computed
    print("First solve (preconditioner setup):")
    x1, info1 = solver.solve(A, b1)
    print(f"  Setup time: {info1.setup_time:.4f}s")
    print(f"  Solve time: {info1.solve_time:.4f}s")
    print(f"  Total: {info1.setup_time + info1.solve_time:.4f}s")
    
    # Subsequent solves - preconditioner is reused
    print("\nSecond solve (preconditioner reused):")
    x2, info2 = solver.solve(A, b2)
    print(f"  Setup time: {info2.setup_time:.4f}s (should be ~0)")
    print(f"  Solve time: {info2.solve_time:.4f}s")
    
    print("\nThird solve (preconditioner reused):")
    x3, info3 = solver.solve(A, b3)
    print(f"  Setup time: {info3.setup_time:.4f}s (should be ~0)")
    print(f"  Solve time: {info3.solve_time:.4f}s")
    print()


def example_callback():
    """Example using callback for monitoring convergence."""
    print("=" * 70)
    print("Example 3: Callback for Monitoring Convergence")
    print("=" * 70)
    
    A = poisson_2d(60, 60)
    b = np.random.rand(A.shape[0])
    
    residuals = []
    
    def callback(xk):
        """Callback function called each iteration."""
        r = b - A @ xk
        res_norm = np.linalg.norm(r)
        residuals.append(res_norm)
        if len(residuals) % 10 == 0:
            print(f"  Iteration {len(residuals)}: residual = {res_norm:.2e}")
    
    x, info = solve(
        A, b,
        backend="scipy",
        method="bicgstab",
        preconditioner="ilu",
        callback=callback
    )
    
    print(f"\nTotal iterations: {len(residuals)}")
    print(f"Final residual: {residuals[-1]:.2e}")
    print()


def example_user_defined_preconditioner():
    """Example using a user-defined preconditioner."""
    print("=" * 70)
    print("Example 4: User-Defined Preconditioner")
    print("=" * 70)
    
    A = poisson_2d(50, 50)
    b = np.random.rand(A.shape[0])
    
    # Define a simple diagonal preconditioner (like Jacobi)
    def my_preconditioner(x):
        """Simple diagonal preconditioner."""
        D_inv = 1.0 / A.diagonal()
        return D_inv * x
    
    # Use as callable preconditioner
    x, info = solve(
        A, b,
        backend="scipy",
        method="bicgstab",
        preconditioner=my_preconditioner
    )
    
    print(f"Converged: {info.converged}")
    print(f"Iterations: {info.iterations}")
    print()


def example_async_solve():
    """Example using async solve."""
    print("=" * 70)
    print("Example 5: Asynchronous Solve")
    print("=" * 70)
    
    A = poisson_2d(100, 100)
    b = np.random.rand(A.shape[0])
    
    solver = AdvancedLinearSolver(
        method="bicgstab",
        backend="scipy",
        preconditioner="ilu"
    )
    
    # Start async solve
    handle = solver.solve_async(A, b)
    
    print("Solve started asynchronously...")
    print("Doing other work...")
    
    # Check status
    while not handle.is_done():
        print("  Still solving...")
        import time
        time.sleep(0.1)
    
    # Get result
    x, info = handle.wait()
    
    print(f"\nSolve completed!")
    print(f"Converged: {info.converged}")
    print(f"Iterations: {info.iterations}")
    print()


def example_backend_selection():
    """Example showing backend selection."""
    print("=" * 70)
    print("Example 6: Backend Selection")
    print("=" * 70)
    
    print("Available backends:")
    backends = BackendRegistry.list_backends()
    for backend in backends:
        print(f"  - {backend}")
    
    print("\nAuto-selected backend:")
    try:
        selected = BackendRegistry.auto_select(prefer_gpu=False)
        print(f"  {selected}")
    except Exception as e:
        print(f"  Error: {e}")
    print()


def example_tolerance_control():
    """Example showing tolerance control."""
    print("=" * 70)
    print("Example 7: Tolerance and Iteration Control")
    print("=" * 70)
    
    A = poisson_2d(80, 80)
    b = np.random.rand(A.shape[0])
    
    # Different tolerance settings
    configs = [
        {"rtol": 1e-6, "atol": 0.0, "name": "rtol=1e-6"},
        {"rtol": 1e-8, "atol": 0.0, "name": "rtol=1e-8"},
        {"rtol": 1e-10, "atol": 0.0, "name": "rtol=1e-10"},
    ]
    
    print(f"{'Configuration':<20} {'Iterations':<12} {'Time (s)':<12} {'Residual':<15}")
    print("-" * 70)
    
    for config in configs:
        x, info = solve(
            A, b,
            backend="scipy",
            method="bicgstab",
            preconditioner="ilu",
            **{k: v for k, v in config.items() if k != "name"}
        )
        print(f"{config['name']:<20} {info.iterations:<12} {info.solve_time:<12.4f} {info.residual_norm:<15.2e}")
    print()


if __name__ == "__main__":
    try:
        example_enhanced_solve()
        example_advanced_solver_class()
        example_callback()
        example_user_defined_preconditioner()
        example_async_solve()
        example_backend_selection()
        example_tolerance_control()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


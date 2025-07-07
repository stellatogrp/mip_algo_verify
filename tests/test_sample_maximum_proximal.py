import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Make tqdm optional
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None):
        print(f"{desc}..." if desc else "Processing...")
        return iterable

from mipalgover.verifier import Verifier

np.set_printoptions(precision=5, suppress=True)


def test_sample_maximum_proximal_point():
    """
    Sample Maximum (SM) methodology for Proximal Point:
    
    1. Sample N = 10,000 problems from parameter space X
    2. Run Proximal Point for k = 1, ..., K iterations on each sample
    3. Compute maximum l∞-norm of fixed-point residuals across all samples
    4. Compare with Verifier bounds (theoretical worst-case)
    
    The sample maximum should be ≤ verifier bound (lower bound property).
    """
    print("="*70)
    print("SAMPLE MAXIMUM (SM) vs VERIFIER COMPARISON")
    print("Proximal Point on Quadratic Problems with Box Constraints")
    print("="*70)
    
    # Problem setup
    n = 4  # Dimension
    K = 11  # Number of iterations to test (matching test_proximal_point.py)
    N = 10000  # Number of samples
    lambd = 1.0  # Proximal parameter
    
    np.random.seed(42)
    
    # Fixed problem structure
    M = np.random.randn(n, n)
    P = M.T @ M + 0.5 * np.eye(n)  # Fixed P matrix
    
    # Box constraints: 0 ≤ x ≤ 1
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.hstack([np.ones(n), np.zeros(n)])
    
    # Parameter space X: uncertainty in linear cost vector q
    q_center = np.random.randn(n)
    q_radius = 10  # Uncertainty radius
    
    print(f"Problem dimension: n = {n}")
    print(f"Iterations: K = {K}")
    print(f"Samples: N = {N}")
    print(f"Proximal parameter: λ = {lambd}")
    print(f"Parameter space: q ∈ [{q_center - q_radius}, {q_center + q_radius}]")
    print(f"Condition number: {np.linalg.cond(P):.2f}")
    
    # SAMPLE MAXIMUM COMPUTATION
    print(f"\nRunning Sample Maximum with {N} samples...")
    
    sample_residuals = []  # Store max residual from each sample
    all_residuals_by_iteration = [[] for _ in range(K)]  # Store residuals by iteration
    
    for sample_idx in tqdm(range(N), desc="Sampling"):
        # Sample random q from parameter space
        q_sample = q_center + q_radius * (2 * np.random.rand(n) - 1)  # Uniform in [-radius, +radius]
        
        # Run Proximal Point on this sample
        x = 0.5 * np.ones(n)  # Start in middle of feasible region
        sample_max_residual = 0.0
        
        for k in range(K):
            # Solve proximal point subproblem using CVXPY
            z = cp.Variable(n)
            obj = 0.5 * cp.quad_form(z, P) + q_sample.T @ z + \
                  1/(2*lambd) * cp.sum_squares(z - x)
            constraints = [A @ z <= b]
            prob = cp.Problem(cp.Minimize(obj), constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            
            # Update iterate
            x_prev = x.copy()
            x = z.value
            
            # Compute fixed-point residual
            residual = np.linalg.norm(x - x_prev, np.inf)
            all_residuals_by_iteration[k].append(residual)
            sample_max_residual = max(sample_max_residual, residual)
        
        sample_residuals.append(sample_max_residual)
    
    # Compute sample maximum statistics
    sample_maximum = np.max(sample_residuals)
    sample_mean = np.mean(sample_residuals)
    sample_95th = np.percentile(sample_residuals, 95)
    
    # Compute sample maximum by iteration
    sample_max_by_iteration = [np.max(residuals) for residuals in all_residuals_by_iteration]
    
    print(f"\nSample Maximum Results:")
    print(f"  Overall sample maximum: {sample_maximum:.6f}")
    print(f"  Sample mean: {sample_mean:.6f}")
    print(f"  Sample 95th percentile: {sample_95th:.6f}")
    
    # VERIFIER COMPUTATION
    print(f"\nRunning Verifier (theoretical worst-case bounds)...")
    
    solver_params = {'OutputFlag': 0, 'MIPGap': 0.001}  # 0.1% optimality gap
    VP = Verifier(solver_params=solver_params)
    
    # Add parameter with uncertainty
    q_param = VP.add_param(n, lb=q_center - q_radius, ub=q_center + q_radius)
    
    # Add initial iterate
    z0 = VP.add_initial_iterate(n, lb=0.5, ub=0.5)  # Start in middle of feasible region
    z = [None for _ in range(K + 1)]
    z[0] = z0
    
    verifier_bounds = []
    for k in range(1, K + 1):
        print(f"  Verifier iteration {k}/{K}...")
        z[k] = VP.proximal_point_step(z[k-1], P, q_param, A, b, lambd, M=10)
        
        VP.set_infinity_norm_objective([z[k] - z[k-1]])
        bound = VP.solve()
        verifier_bounds.append(bound)
        
        print(f"    Iteration {k}: Sample max = {sample_max_by_iteration[k-1]:.6f}, "
              f"Verifier bound = {bound:.6f}")
    
    verifier_maximum = np.max(verifier_bounds)
    
    print(f"\nVerifier Results:")
    print(f"  Maximum verifier bound: {verifier_maximum:.6f}")
    print(f"  Final iteration bound: {verifier_bounds[-1]:.6f}")
    
    # COMPARISON AND VALIDATION
    print(f"\n" + "="*50)
    print("COMPARISON:")
    print(f"  Sample Maximum:     {sample_maximum:.6f}")
    print(f"  Verifier Bound:     {verifier_maximum:.6f}")
    print(f"  Gap (Verifier - SM): {verifier_maximum - sample_maximum:.6f}")
    print(f"  Ratio (SM/Verifier): {sample_maximum / verifier_maximum:.3f}")
    
    # Validation: Sample maximum should be ≤ Verifier bound
    gap_by_iteration = np.array(verifier_bounds) - np.array(sample_max_by_iteration)
    all_valid = np.all(gap_by_iteration >= -1e-6)  # Allow small numerical tolerance
    
    print(f"  Valid lower bound: {all_valid}")
    if not all_valid:
        invalid_iterations = np.where(gap_by_iteration < -1e-6)[0] + 1
        print(f"  Invalid iterations: {invalid_iterations}")
    
    print("="*50)
    
    # PLOTTING
    plt.figure(figsize=(10, 6))
    
    # Residuals comparison plot
    iterations = range(1, K + 1)
    plt.semilogy(iterations, sample_max_by_iteration, 'b-', label='Sample Maximum', linewidth=2)
    plt.semilogy(iterations, verifier_bounds, 'r--', label='Verifier Bounds', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Max l∞ Fixed-point Residual')
    plt.title(f'Sample Maximum vs Verifier Bounds\n(n={n}, K={K}, N={N} samples, λ={lambd})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ASSERTIONS
    assert all_valid, "Sample maximum should be a lower bound on verifier bounds"
    assert sample_maximum <= verifier_maximum + 1e-6, "Overall sample max should be ≤ verifier max"
    
    print("✓ All validations passed! Sample Maximum is a valid lower bound.")
    
    return {
        'sample_maximum': sample_maximum,
        'verifier_maximum': verifier_maximum,
        'sample_max_by_iteration': sample_max_by_iteration,
        'verifier_bounds': verifier_bounds,
        'gap': verifier_maximum - sample_maximum,
        'valid': all_valid
    }


if __name__ == "__main__":
    # Run Proximal Point sample maximum test
    pp_results = test_sample_maximum_proximal_point() 
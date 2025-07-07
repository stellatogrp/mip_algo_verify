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
  
    print("="*70)
    print("SAMPLE MAXIMUM (SM) vs VERIFIER COMPARISON")
    print("Proximal Point on Quadratic Problems with Box Constraints")
    print("="*70)
    
    
    n = 4 
    K = 11  
    N = 10000  
    lambd = 1.0  
    
    np.random.seed(42)
    
   
    M = np.random.randn(n, n)
    P = M.T @ M + 0.5 * np.eye(n)  
    
    
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.hstack([np.ones(n), np.zeros(n)])
    
    
    q_center = np.random.randn(n)
    q_radius = 10  
    
    print(f"Problem dimension: n = {n}")
    print(f"Iterations: K = {K}")
    print(f"Samples: N = {N}")
    print(f"Proximal parameter: λ = {lambd}")
    print(f"Parameter space: q ∈ [{q_center - q_radius}, {q_center + q_radius}]")
    print(f"Condition number: {np.linalg.cond(P):.2f}")
    
   
    print(f"\nRunning Sample Maximum with {N} samples...")
    
    sample_residuals = []  
    all_residuals_by_iteration = [[] for _ in range(K)]  
    
    for sample_idx in tqdm(range(N), desc="Sampling"):
        
        q_sample = q_center + q_radius * (2 * np.random.rand(n) - 1)  
        
       
        x = 0.5 * np.ones(n) 
        sample_max_residual = 0.0
        
        for k in range(K):
           
            z = cp.Variable(n)
            obj = 0.5 * cp.quad_form(z, P) + q_sample.T @ z + \
                  1/(2*lambd) * cp.sum_squares(z - x)
            constraints = [A @ z <= b]
            prob = cp.Problem(cp.Minimize(obj), constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            
           
            x_prev = x.copy()
            x = z.value
            
            residual = np.linalg.norm(x - x_prev, np.inf)
            all_residuals_by_iteration[k].append(residual)
            sample_max_residual = max(sample_max_residual, residual)
        
        sample_residuals.append(sample_max_residual)
    
  
    sample_maximum = np.max(sample_residuals)
    sample_mean = np.mean(sample_residuals)
    sample_95th = np.percentile(sample_residuals, 95)
    
   
    sample_max_by_iteration = [np.max(residuals) for residuals in all_residuals_by_iteration]
    
    print(f"\nSample Maximum Results:")
    print(f"  Overall sample maximum: {sample_maximum:.6f}")
    print(f"  Sample mean: {sample_mean:.6f}")
    print(f"  Sample 95th percentile: {sample_95th:.6f}")
    
   
    print(f"\nRunning Verifier (theoretical worst-case bounds)...")
    
    solver_params = {'OutputFlag': 0, 'MIPGap': 0.001}  
    VP = Verifier(solver_params=solver_params)
    
   
    q_param = VP.add_param(n, lb=q_center - q_radius, ub=q_center + q_radius)
    
   
    z0 = VP.add_initial_iterate(n, lb=0.5, ub=0.5)  
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
    
   
    print(f"\n" + "="*50)
    print("COMPARISON:")
    print(f"  Sample Maximum:     {sample_maximum:.6f}")
    print(f"  Verifier Bound:     {verifier_maximum:.6f}")
    print(f"  Gap (Verifier - SM): {verifier_maximum - sample_maximum:.6f}")
    print(f"  Ratio (SM/Verifier): {sample_maximum / verifier_maximum:.3f}")
    
  
    gap_by_iteration = np.array(verifier_bounds) - np.array(sample_max_by_iteration)
    all_valid = np.all(gap_by_iteration >= -1e-6)
    
    print(f"  Valid lower bound: {all_valid}")
    if not all_valid:
        invalid_iterations = np.where(gap_by_iteration < -1e-6)[0] + 1
        print(f"  Invalid iterations: {invalid_iterations}")
    
    print("="*50)
    
   
    plt.figure(figsize=(10, 6))
    
   
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
    
    pp_results = test_sample_maximum_proximal_point() 

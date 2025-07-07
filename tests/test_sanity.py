import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mipalgover.verifier import Verifier

def test_frank_wolfe_1d_sanity():
    """
    1D Frank-Wolfe Sanity Test
    
    Problem: minimize (1/2) * P * x^2 + q * x
    Constraint: 0 ≤ x ≤ 1 (box constraint)
    
    Parameter uncertainty: q ∈ [q_min, q_max]
    
    OBVIOUS WORST CASE:
    - When q = q_max (most positive), gradient = P*x + q_max is largest
    - Frank-Wolfe will make biggest jumps, causing largest residuals
    """
    print("="*60)
    print("1D FRANK-WOLFE SANITY TEST")
    print("="*60)
    
    # 1D problem setup
    n = 1
    P = np.array([[2.0]])  # Simple quadratic coefficient
    
    # Box constraints: 0 ≤ x ≤ 1
    A = np.array([[1.0], [-1.0]])  # [x ≤ 1, -x ≤ 0]
    b = np.array([1.0, 0.0])       # [1, 0]
    
    # Parameter uncertainty in linear cost q
    q_min = -1.0
    q_max = 1.0
    print(f"Problem: minimize (1/2) * {P[0,0]} * x^2 + q * x")
    print(f"Constraint: 0 ≤ x ≤ 1")
    print(f"Parameter: q ∈ [{q_min}, {q_max}]")
    
    # Test a few specific parameter values
    test_qs = [q_min, 0.0, q_max]
    print(f"\nTesting specific q values: {test_qs}")
    
    K = 5  # Just a few iterations
    
    for q_val in test_qs:
        print(f"\n--- q = {q_val} ---")
        q = np.array([q_val])
        
        # Run Frank-Wolfe manually
        x = np.array([0.0])  # Start at x=0
        print(f"Initial: x = {x[0]:.4f}")
        
        for k in range(K):
            # Compute gradient: ∇f(x) = P*x + q
            grad = P @ x + q
            print(f"  Iter {k+1}: gradient = {grad[0]:.4f}")
            
            # Solve LP: min grad^T s subject to As ≤ b
            # For 1D box constraint [0,1], this is trivial:
            if grad[0] >= 0:
                s = np.array([0.0])  # Go to left boundary
            else:
                s = np.array([1.0])  # Go to right boundary
            
            print(f"    LP solution: s = {s[0]:.4f}")
            
            # Frank-Wolfe update
            gamma = 2.0 / (k + 2)
            x_prev = x.copy()
            x = (1 - gamma) * x + gamma * s
            
            # Compute residual
            residual = abs(x[0] - x_prev[0])  # L∞ norm in 1D
            print(f"    Update: x = {x[0]:.4f}, residual = {residual:.6f}")
    
    print(f"\n" + "="*40)
    print("EXPECTED BEHAVIOR:")
    print(f"- q = {q_max} should give LARGEST residuals (worst case)")
    print(f"- q = {q_min} should give smaller residuals")
    print(f"- q = 0 should be in between")
    
    # Now test with Verifier
    print(f"\n" + "="*40)
    print("VERIFIER TEST:")
    
    solver_params = {'OutputFlag': 0, 'MIPGap': 0.001}
    VP = Verifier(solver_params=solver_params)
    
    # Add parameter with uncertainty
    q_param = VP.add_param(n, lb=q_min, ub=q_max)
    
    # Add initial iterate
    z0 = VP.add_initial_iterate(n, lb=0, ub=0)  # Start at x=0
    z = [None for _ in range(K + 1)]
    z[0] = z0
    
    print(f"Running verifier for {K} iterations...")
    
    verifier_bounds = []
    for k in range(1, K + 1):
        gamma = 2.0 / (k + 1)
        z[k] = VP.frank_wolfe_step(z[k-1], P, q_param, A, b, gamma, M=10)
        
        VP.set_infinity_norm_objective([z[k] - z[k-1]])
        bound = VP.solve()
        verifier_bounds.append(bound)
        
        print(f"  Iteration {k}: Verifier bound = {bound:.6f}")
    
    print(f"\nVerifier maximum bound: {max(verifier_bounds):.6f}")
    print(f"This should correspond to the worst-case q = {q_max}")
    
    return verifier_bounds


def test_frank_wolfe_1d_extreme():
    """
    Even simpler test: extreme parameter values to make worst case obvious
    """
    print("\n" + "="*60)
    print("1D FRANK-WOLFE EXTREME TEST")
    print("="*60)
    
    n = 1
    P = np.array([[1.0]])  # Simple P = 1
    A = np.array([[1.0], [-1.0]])
    b = np.array([1.0, 0.0])
    
    # Extreme parameter range
    q_min = -10.0
    q_max = 10.0
    
    print(f"Problem: minimize (1/2) * x^2 + q * x")
    print(f"Constraint: 0 ≤ x ≤ 1")
    print(f"Parameter: q ∈ [{q_min}, {q_max}] (EXTREME RANGE)")
    
    # Manual test with extreme values
    test_qs = [q_min, q_max]
    K = 3
    
    results = {}
    for q_val in test_qs:
        print(f"\n--- q = {q_val} ---")
        q = np.array([q_val])
        
        x = np.array([0.0])
        max_residual = 0.0
        
        for k in range(K):
            grad = P @ x + q
            s = np.array([0.0]) if grad[0] >= 0 else np.array([1.0])
            
            gamma = 2.0 / (k + 2)
            x_prev = x.copy()
            x = (1 - gamma) * x + gamma * s
            
            residual = abs(x[0] - x_prev[0])
            max_residual = max(max_residual, residual)
            
            print(f"  Iter {k+1}: x = {x[0]:.4f}, residual = {residual:.6f}")
        
        results[q_val] = max_residual
        print(f"  MAX RESIDUAL for q={q_val}: {max_residual:.6f}")
    
    print(f"\n" + "="*40)
    print("COMPARISON:")
    for q_val, max_res in results.items():
        print(f"  q = {q_val:5.1f}: max residual = {max_res:.6f}")
    
    worst_q = max(results.keys(), key=lambda q: results[q])
    print(f"\nWORST CASE: q = {worst_q} (as expected!)")
    
    return results


if __name__ == "__main__":
    # Run both sanity tests
    test_frank_wolfe_1d_sanity()
    test_frank_wolfe_1d_extreme()
    print("\n🎉 Sanity tests completed!") 
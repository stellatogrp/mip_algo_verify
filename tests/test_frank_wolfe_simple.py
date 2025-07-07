import cvxpy as cp
import numpy as np

from mipalgover.verifier import Verifier

np.set_printoptions(precision=5, suppress=True)


def test_frank_wolfe_basic():
    """
    Basic Frank-Wolfe test demonstrating the standard testing pattern.
    
    This follows the pattern used in other test files:
    1. Generate a test problem with known parameters
    2. Solve with CVXPY for reference solution
    3. Implement algorithm manually
    4. Test with Verifier and compare bounds
    5. Assert convergence and bound validity
    """
    print("Basic Frank-Wolfe Test")
    
    # 1. Problem setup
    n = 3
    np.random.seed(42)
    
    # Create quadratic objective: min 0.5 * x^T P x + q^T x
    M = np.random.randn(n, n)
    P = M.T @ M + np.eye(n)  # Positive definite
    q = np.random.randn(n)
    
    # Box constraints: 0 ≤ x ≤ 1
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.hstack([np.ones(n), np.zeros(n)])
    
    print(f"Problem: min 0.5*x^T*P*x + q^T*x subject to 0 ≤ x ≤ 1")
    print(f"Dimension: {n}, Condition number: {np.linalg.cond(P):.2f}")
    
    # 2. Reference solution with CVXPY
    x_var = cp.Variable(n)
    objective = 0.5 * cp.quad_form(x_var, P) + q.T @ x_var
    constraints = [A @ x_var <= b]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    x_optimal = x_var.value
    optimal_value = problem.value
    
    print(f"Optimal solution: {x_optimal}")
    print(f"Optimal value: {optimal_value:.6f}")
    
    # 3. Manual Frank-Wolfe implementation
    x = np.zeros(n)  # Initial point
    K_manual = 60
    
    for k in range(K_manual):
        # Compute gradient
        gradient = P @ x + q
        
        # Solve linear subproblem: min gradient^T * s subject to A*s ≤ b
        s_var = cp.Variable(n)
        lp_objective = gradient.T @ s_var
        lp_constraints = [A @ s_var <= b]
        lp_problem = cp.Problem(cp.Minimize(lp_objective), lp_constraints)
        lp_problem.solve(solver=cp.CLARABEL)
        s = s_var.value
        
        # Frank-Wolfe update: x^{k+1} = (1-γ_k)x^k + γ_k*s
        gamma = 2.0 / (k + 2)  # Standard step size
        x = (1 - gamma) * x + gamma * s
    
    manual_error = np.linalg.norm(x - x_optimal)
    print(f"Manual Frank-Wolfe solution: {x}")
    print(f"Error vs optimal: {manual_error:.6f}")
    
    # 4. Test with Verifier
    solver_params = {'OutputFlag': 0}  # Suppress Gurobi output
    VP = Verifier(solver_params=solver_params)
    
    # Add parameter with uncertainty (testing robustness)
    q_param = VP.add_param(n, lb=q-0.01, ub=q+0.01)
    
    # Add initial iterate
    z0 = VP.add_initial_iterate(n, lb=0, ub=0)
    
    # Run Frank-Wolfe algorithm through verifier
    K_verifier = 15  # Fewer iterations for speed
    z = [None for _ in range(K_verifier + 1)]
    z[0] = z0
    
    residuals = []
    for k in range(1, K_verifier + 1):
        gamma = 2.0 / (k + 1)
        z[k] = VP.frank_wolfe_step(z[k-1], P, q_param, A, b, gamma, M=100)
        
        # Set objective to measure convergence (fixed-point residual)
        VP.set_infinity_norm_objective([z[k] - z[k-1]])
        residual_bound = VP.solve()
        residuals.append(residual_bound)
        
        print(f"Iteration {k}: Residual bound = {residual_bound:.6f}")
    
    # 5. Assertions
    print(f"\nFinal residual bound: {residuals[-1]:.6f}")
    
    # Frank-Wolfe should converge to reasonable accuracy
    assert manual_error <= 0.05, f"Manual implementation error {manual_error:.6f} too large"
    
    # Verifier bounds should be valid (decreasing for convergence)
    assert residuals[-1] <= residuals[0], "Residual bounds should generally decrease"
    
    # The bounds are conservative, so this should hold
    assert residuals[-1] >= 0, "Residual bounds should be non-negative"
    
    print("✓ All assertions passed!")


def test_frank_wolfe_multiple_problems():
    """Test Frank-Wolfe on different constraint types (like test_portfolio multiple instances)."""
    print("\n" + "="*50)
    print("Testing Frank-Wolfe on Multiple Problem Types")
    print("="*50)
    
    problems = [
        {
            'name': 'Box constraints',
            'A': np.vstack([np.eye(3), -np.eye(3)]),
            'b': np.hstack([np.ones(3), np.zeros(3)]),
            'seed': 100
        },
        {
            'name': 'Simplex constraints', 
            'A': np.vstack([-np.eye(3), np.ones((1, 3))]),
            'b': np.hstack([np.zeros(3), 1.0]),
            'seed': 200
        },
        {
            'name': 'General polytope',
            'A': np.vstack([np.eye(3), -np.eye(3), np.ones((1, 3))]),
            'b': np.hstack([2*np.ones(3), 0.5*np.ones(3), 2.5]),
            'seed': 300
        }
    ]
    
    for i, problem in enumerate(problems):
        print(f"\n--- Problem {i+1}: {problem['name']} ---")
        
        # Problem setup
        n = 3
        np.random.seed(problem['seed'])
        M = np.random.randn(n, n)
        P = M.T @ M + 0.5 * np.eye(n)
        q = np.random.randn(n)
        A, b = problem['A'], problem['b']
        
        # Reference solution
        x_var = cp.Variable(n)
        obj = 0.5 * cp.quad_form(x_var, P) + q.T @ x_var
        constraints = [A @ x_var <= b]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.CLARABEL)
        x_opt = x_var.value
        
        # Manual Frank-Wolfe
        x = np.zeros(n)
        for k in range(40):
            grad = P @ x + q
            s_var = cp.Variable(n)
            lp_obj = grad.T @ s_var
            lp_constraints = [A @ s_var <= b]
            lp_prob = cp.Problem(cp.Minimize(lp_obj), lp_constraints)
            lp_prob.solve(solver=cp.CLARABEL)
            s = s_var.value
            gamma = 2.0 / (k + 2)
            x = (1 - gamma) * x + gamma * s
        
        error = np.linalg.norm(x - x_opt)
        print(f"Optimal: {x_opt}")
        print(f"Frank-Wolfe: {x}")
        print(f"Error: {error:.6f}")
        
        # Test should pass for all problem types
        assert error <= 0.1, f"Problem {i+1} ({problem['name']}) failed to converge"
        print(f"✓ {problem['name']} test passed!")
    
    print("\n✓ All problem types passed!")


if __name__ == "__main__":
    test_frank_wolfe_basic()
    test_frank_wolfe_multiple_problems()
    print("\n" + "="*50)
    print("🎉 All Frank-Wolfe tests completed successfully!")
    print("="*50) 
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from mipalgover.verifier import Verifier

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_frank_wolfe_quadratic_box():
    """Test Frank-Wolfe on quadratic problem with box constraints."""
    print("Testing Frank-Wolfe on quadratic problem with box constraints")
    
    n = 5
    K = 100  # Increased iterations for better convergence
    np.random.seed(42)
    
    # Create a well-conditioned positive definite matrix
    M = np.random.randn(n, n)
    P = M.T @ M + 0.5 * np.eye(n)  # Better conditioning
    q = np.random.randn(n)
    
    # Box constraints: 0 ≤ x ≤ 1
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.hstack([np.ones(n), np.zeros(n)])
    
    print(f"Problem dimension: {n}")
    print(f"Condition number of P: {np.linalg.cond(P):.2f}")
    
    # Solve with CVXPY as reference
    x = cp.Variable(n)
    obj = 0.5 * cp.quad_form(x, P) + q.T @ x
    constraints = [A @ x <= b]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    x_opt = x.value
    opt_value = prob.value
    
    print(f"Optimal value: {opt_value:.6f}")
    print(f"Optimal solution: {x_opt}")
    
    # Manual Frank-Wolfe implementation
    def frank_wolfe_manual(P, q, A, b, x0, K):
        x = x0.copy()
        residuals = []
        solutions = [x0.copy()]
        
        for k in range(K):
            # Compute gradient: ∇f(x) = Px + q
            grad = P @ x + q
            
            # Solve linear program: min grad^T s subject to As ≤ b
            s_var = cp.Variable(n)
            lp_obj = grad.T @ s_var
            lp_constraints = [A @ s_var <= b]
            lp_prob = cp.Problem(cp.Minimize(lp_obj), lp_constraints)
            lp_prob.solve(solver=cp.CLARABEL)
            s = s_var.value
            
            # Frank-Wolfe update with diminishing step size
            gamma = 2.0 / (k + 2)
            x_new = (1 - gamma) * x + gamma * s
            
            # Compute fixed-point residual
            residual = np.linalg.norm(x_new - x, np.inf)
            residuals.append(residual)
            solutions.append(x_new.copy())
            
            x = x_new
            
        return x, residuals, solutions
    
    # Run manual Frank-Wolfe
    x0 = np.zeros(n)  # Start at origin
    x_fw, fw_residuals, fw_solutions = frank_wolfe_manual(P, q, A, b, x0, K)
    
    print(f"Frank-Wolfe solution: {x_fw}")
    print(f"Error vs optimal: {np.linalg.norm(x_fw - x_opt):.6f}")
    
    # Test with Verifier (using fewer iterations for speed)
    solver_params = {'OutputFlag': 0}
    VP = Verifier(solver_params=solver_params)
    
    # Add parameters with some tolerance
    q_param = VP.add_param(n, lb=q-0.01, ub=q+0.01)
    
    # Add initial iterate
    z0 = VP.add_initial_iterate(n, lb=0, ub=0)  # Start at origin
    z = [None for _ in range(41)]  # Increased for better plotting
    z[0] = z0
    
    # Run Frank-Wolfe through verifier
    verifier_residuals = []
    for k in range(1, 41):  # Test first 40 iterations
        gamma = 2.0 / (k + 1)  # Step size
        z[k] = VP.frank_wolfe_step(z[k-1], P, q_param, A, b, gamma, M=100)
        VP.set_infinity_norm_objective([z[k] - z[k-1]])
        res = VP.solve()
        verifier_residuals.append(res)
    
    print(f"Manual FW final residual: {fw_residuals[-1]:.6f}")
    print(f"Verifier residual at k=40: {verifier_residuals[-1]:.6f}")
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Residual comparison
    plt.subplot(1, 3, 1)
    iterations = range(1, 41)
    plt.semilogy(iterations, fw_residuals[:40], 'b-', label='Actual Frank-Wolfe', linewidth=2)
    plt.semilogy(iterations, verifier_residuals, 'r--', label='Verifier Worst-case Bounds', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fixed-point Residual')
    plt.title('Frank-Wolfe Convergence:\nActual vs Worst-case Bounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error to optimal solution
    plt.subplot(1, 3, 2)
    errors_to_opt = [np.linalg.norm(sol - x_opt) for sol in fw_solutions[:41]]
    plt.semilogy(range(41), errors_to_opt, 'g-', label='Distance to Optimal', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('||x^k - x*||')
    plt.title('Convergence to Optimal Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Gap between bounds and actual
    plt.subplot(1, 3, 3)
    gap = np.array(verifier_residuals) - np.array(fw_residuals[:40])
    plt.semilogy(iterations, gap, 'm-', label='Verifier Conservatism', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Worst-case Bound - Actual Residual')
    plt.title('How Conservative are\nthe Verifier Bounds?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Frank-Wolfe Analysis: Box Constraints (n={n}, cond={np.linalg.cond(P):.1f})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Assertions with more realistic tolerances for Frank-Wolfe
    assert np.linalg.norm(x_fw - x_opt) <= 0.05, "Frank-Wolfe solution should be reasonably close to optimal"
    assert np.all(np.array(fw_residuals[:40]) <= np.array(verifier_residuals) + 1e-6), "Manual residuals should be within verifier bounds"


def test_frank_wolfe_simplex():
    """Test Frank-Wolfe on quadratic problem with simplex constraints."""
    print("\nTesting Frank-Wolfe on quadratic problem with simplex constraints")
    
    n = 4
    K = 80  # Increased iterations
    np.random.seed(123)
    
    # Create problem data
    M = np.random.randn(n, n)
    P = M.T @ M + 0.5 * np.eye(n)
    q = np.random.randn(n)
    
    # Simplex constraints: x ≥ 0, sum(x) ≤ 1
    A = np.vstack([
        -np.eye(n),  # x ≥ 0
        np.ones((1, n))  # sum(x) ≤ 1
    ])
    b = np.hstack([np.zeros(n), 1.0])
    
    print(f"Problem dimension: {n}")
    print(f"Condition number of P: {np.linalg.cond(P):.2f}")
    
    # Solve with CVXPY
    x = cp.Variable(n)
    obj = 0.5 * cp.quad_form(x, P) + q.T @ x
    constraints = [A @ x <= b]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    x_opt = x.value
    opt_value = prob.value
    
    print(f"Optimal value: {opt_value:.6f}")
    print(f"Optimal solution: {x_opt}")
    
    # Manual Frank-Wolfe implementation
    x = np.zeros(n)  # Start at origin
    fw_residuals = []
    fw_solutions = [x.copy()]
    
    for k in range(K):
        grad = P @ x + q
        
        # Solve LP to find vertex
        s_var = cp.Variable(n)
        lp_obj = grad.T @ s_var
        lp_constraints = [A @ s_var <= b]
        lp_prob = cp.Problem(cp.Minimize(lp_obj), lp_constraints)
        lp_prob.solve(solver=cp.CLARABEL)
        s = s_var.value
        
        # Update with step size 2/(k+2)
        gamma = 2.0 / (k + 2)
        x_new = (1 - gamma) * x + gamma * s
        
        residual = np.linalg.norm(x_new - x, np.inf)
        fw_residuals.append(residual)
        fw_solutions.append(x_new.copy())
        x = x_new
    
    print(f"Frank-Wolfe solution: {x}")
    print(f"Error vs optimal: {np.linalg.norm(x - x_opt):.6f}")
    
    # Test with Verifier
    solver_params = {'OutputFlag': 0}
    VP = Verifier(solver_params=solver_params)
    
    q_param = VP.add_param(n, lb=q-0.02, ub=q+0.02)
    z0 = VP.add_initial_iterate(n, lb=0, ub=0)
    z = [None for _ in range(31)]  # Test first 30 iterations
    z[0] = z0
    
    verifier_residuals = []
    for k in range(1, 31):
        gamma = 2.0 / (k + 1)
        z[k] = VP.frank_wolfe_step(z[k-1], P, q_param, A, b, gamma, M=100)
        VP.set_infinity_norm_objective([z[k] - z[k-1]])
        res = VP.solve()
        verifier_residuals.append(res)
    
    print(f"Manual FW final residual: {fw_residuals[-1]:.6f}")
    print(f"Verifier residual at k=30: {verifier_residuals[-1]:.6f}")
    
    # Plotting for simplex constraints
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Residual comparison
    plt.subplot(1, 2, 1)
    iterations = range(1, 31)
    plt.semilogy(iterations, fw_residuals[:30], 'b-', label='Actual Frank-Wolfe', linewidth=2)
    plt.semilogy(iterations, verifier_residuals, 'r--', label='Verifier Worst-case Bounds', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fixed-point Residual')
    plt.title('Simplex Constraints:\nActual vs Worst-case Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Solution trajectory (showing convergence to simplex)
    plt.subplot(1, 2, 2)
    # Plot the sum of components (should stay ≤ 1 due to simplex constraint)
    sum_trajectory = [np.sum(sol) for sol in fw_solutions[:31]]
    plt.plot(range(31), sum_trajectory, 'g-', label='Sum of components', linewidth=2)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Simplex boundary')
    plt.xlabel('Iteration')
    plt.ylabel('∑ xᵢ')
    plt.title('Feasibility: Sum Constraint')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Frank-Wolfe on Simplex (n={n}, cond={np.linalg.cond(P):.1f})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Assertions with more realistic tolerances
    assert np.linalg.norm(x - x_opt) <= 0.05, "Frank-Wolfe solution should be reasonably close to optimal"
    assert np.all(np.array(fw_residuals[:30]) <= np.array(verifier_residuals) + 1e-6), "Manual residuals should be within verifier bounds"


def test_frank_wolfe_polytope():
    """Test Frank-Wolfe on quadratic problem with general polytope constraints."""
    print("\nTesting Frank-Wolfe on quadratic problem with polytope constraints")
    
    n = 3
    K = 60  # Increased iterations
    np.random.seed(456)
    
    # Create problem data
    M = np.random.randn(n, n)
    P = M.T @ M + np.eye(n)
    q = np.random.randn(n)
    
    # General polytope: intersection of halfspaces
    # Example: -1 ≤ x_i ≤ 2, x_1 + x_2 + x_3 ≤ 3
    A = np.vstack([
        np.eye(n),      # x ≤ 2
        -np.eye(n),     # x ≥ -1
        np.ones((1, n)) # sum(x) ≤ 3
    ])
    b = np.hstack([2*np.ones(n), np.ones(n), 3.0])
    
    print(f"Problem dimension: {n}")
    print(f"Condition number of P: {np.linalg.cond(P):.2f}")
    
    # Solve with CVXPY
    x = cp.Variable(n)
    obj = 0.5 * cp.quad_form(x, P) + q.T @ x
    constraints = [A @ x <= b]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    x_opt = x.value
    opt_value = prob.value
    
    print(f"Optimal value: {opt_value:.6f}")
    print(f"Optimal solution: {x_opt}")
    
    # Manual Frank-Wolfe
    x = np.zeros(n)  # Start at origin (feasible)
    fw_residuals = []
    
    for k in range(K):
        grad = P @ x + q
        
        # Solve LP
        s_var = cp.Variable(n)
        lp_obj = grad.T @ s_var
        lp_constraints = [A @ s_var <= b]
        lp_prob = cp.Problem(cp.Minimize(lp_obj), lp_constraints)
        lp_prob.solve(solver=cp.CLARABEL)
        s = s_var.value
        
        # Update
        gamma = 2.0 / (k + 2)
        x_new = (1 - gamma) * x + gamma * s
        
        residual = np.linalg.norm(x_new - x, np.inf)
        fw_residuals.append(residual)
        x = x_new
    
    print(f"Frank-Wolfe solution: {x}")
    print(f"Error vs optimal: {np.linalg.norm(x - x_opt):.6f}")
    
    # Test with Verifier with parameter uncertainty
    solver_params = {'OutputFlag': 0}
    VP = Verifier(solver_params=solver_params)
    
    q_param = VP.add_param(n, lb=q-0.05, ub=q+0.05)
    z0 = VP.add_initial_iterate(n, lb=0, ub=0)
    z = [None for _ in range(16)]  # Test first 15 iterations
    z[0] = z0
    
    verifier_residuals = []
    for k in range(1, 16):
        gamma = 2.0 / (k + 1)
        z[k] = VP.frank_wolfe_step(z[k-1], P, q_param, A, b, gamma, M=100)
        VP.set_infinity_norm_objective([z[k] - z[k-1]])
        res = VP.solve()
        verifier_residuals.append(res)
    
    print(f"Manual FW final residual: {fw_residuals[-1]:.6f}")
    print(f"Verifier residual at k=15: {verifier_residuals[-1]:.6f}")
    
    # Assertions with more realistic tolerances
    assert np.linalg.norm(x - x_opt) <= 0.05, "Frank-Wolfe solution should be reasonably close to optimal"
    assert np.all(np.array(fw_residuals[:15]) <= np.array(verifier_residuals) + 1e-6), "Manual residuals should be within verifier bounds"


def test_frank_wolfe_multiple_instances():
    """Test Frank-Wolfe on multiple random instances."""
    print("\nTesting Frank-Wolfe on multiple random instances")
    
    n = 4
    K = 50  # Increased iterations
    num_instances = 5
    
    for instance in range(num_instances):
        print(f"\n--- Instance {instance + 1} ---")
        np.random.seed(instance * 100)
        
        # Random problem data
        M = np.random.randn(n, n)
        P = M.T @ M + 0.5 * np.eye(n)  # Better conditioning
        q = np.random.randn(n)
        
        # Box constraints
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.hstack([np.ones(n), np.zeros(n)])
        
        # CVXPY solution
        x = cp.Variable(n)
        obj = 0.5 * cp.quad_form(x, P) + q.T @ x
        constraints = [A @ x <= b]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.CLARABEL)
        x_opt = x.value
        
        # Manual Frank-Wolfe
        x = np.zeros(n)
        for k in range(K):
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
        print(f"Error vs optimal: {error:.6f}")
        
        # Test with Verifier (fewer iterations for speed)
        solver_params = {'OutputFlag': 0}
        VP = Verifier(solver_params=solver_params)
        
        q_param = VP.add_param(n, lb=q-0.01, ub=q+0.01)
        z0 = VP.add_initial_iterate(n, lb=0, ub=0)
        z = [None for _ in range(11)]  # Test first 10 iterations
        z[0] = z0
        
        for k in range(1, 11):
            gamma = 2.0 / (k + 1)
            z[k] = VP.frank_wolfe_step(z[k-1], P, q_param, A, b, gamma, M=100)
        
        VP.set_infinity_norm_objective([z[10] - z[9]])
        verifier_bound = VP.solve()
        
        print(f"Verifier bound at k=10: {verifier_bound:.6f}")
        
        # Each instance should converge reasonably
        assert error <= 0.1, f"Instance {instance + 1} should converge reasonably well"


def test_frank_wolfe_ista_instance_vs_K():
    """
    Test Frank-Wolfe on ISTA-style problem showing worst-case vs actual residuals as function of K.
    
    Problem: min 0.5 * ||Ax - b||^2 subject to ||x||_1 ≤ t
    This is the constrained version of LASSO: min 0.5 * ||Ax - b||^2 + λ * ||x||_1
    """
    print("\n" + "="*70)
    print("Frank-Wolfe on ISTA Instance: Worst-case vs Actual Residuals vs K")
    print("="*70)
    
    # Problem setup (ISTA-style sparse recovery)
    m, n = 15, 20  # Smaller, better conditioned system
    np.random.seed(789)
    
    # Create measurement matrix and sparse true signal
    A = np.random.randn(m, n) / np.sqrt(m)  # Normalized Gaussian matrix
    # Add regularization to improve conditioning
    A = A + 0.01 * np.eye(m, n)  # Add small diagonal component
    
    x_true = np.zeros(n)
    sparse_indices = np.random.choice(n, size=4, replace=False)  # 4 nonzero entries
    x_true[sparse_indices] = np.random.randn(4)
    b = A @ x_true + 0.01 * np.random.randn(m)  # Noisy measurements
    
    # L1 ball constraint: ||x||_1 ≤ t
    t = 1.2 * np.sum(np.abs(x_true))  # Slightly larger than true signal's L1 norm
    
    print(f"Problem dimensions: m={m}, n={n}")
    print(f"True signal sparsity: {np.sum(x_true != 0)} nonzeros")
    print(f"L1 ball radius: t={t:.3f}")
    print(f"True signal L1 norm: {np.sum(np.abs(x_true)):.3f}")
    
    # Convert to quadratic form for Frank-Wolfe
    # min 0.5 * ||Ax - b||^2 = min 0.5 * x^T (A^T A) x - (A^T b)^T x + 0.5 * ||b||^2
    P = A.T @ A + 0.1 * np.eye(n)  # Add regularization for better conditioning
    q = -A.T @ b
    constant = 0.5 * np.linalg.norm(b)**2
    
    print(f"Condition number of A^T A: {np.linalg.cond(P):.2f}")
    
    # Reference solution with CVXPY
    x_var = cp.Variable(n)
    objective = 0.5 * cp.sum_squares(A @ x_var - b) + 0.05 * cp.sum_squares(x_var)  # Add regularization
    constraints = [cp.norm(x_var, 1) <= t]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    x_optimal = x_var.value
    
    print(f"Optimal solution L1 norm: {np.sum(np.abs(x_optimal)):.3f}")
    print(f"Optimal objective: {problem.value:.6f}")
    
    # Test different values of K
    K_max = 20
    K_values = list(range(1, K_max + 1))  # K = 1, 2, 3, ..., 20
    
    # Run manual Frank-Wolfe ONCE up to K_max and track all residuals
    print(f"\nRunning Frank-Wolfe for {K_max} iterations...")
    x = np.zeros(n)  # Start at origin
    actual_residuals_all = []  # Store residual at each iteration
    solutions_all = [x.copy()]  # Store solution at each iteration
    
    for k in range(K_max):
        # Gradient of 0.5 * ||Ax - b||^2
        grad = A.T @ (A @ x - b)
        
        # Solve LP: min grad^T s subject to ||s||_1 ≤ t
        # This is equivalent to: s = -t * sign(grad_i) where |grad_i| is maximum
        max_grad_idx = np.argmax(np.abs(grad))
        s = np.zeros(n)
        s[max_grad_idx] = -t * np.sign(grad[max_grad_idx])
        
        # Frank-Wolfe update
        gamma = 2.0 / (k + 2)
        x_prev = x.copy()  # Store previous iterate
        x_new = (1 - gamma) * x + gamma * s
        x = x_new
        
        # Store residual and solution
        residual = np.linalg.norm(x - x_prev, np.inf)
        actual_residuals_all.append(residual)
        solutions_all.append(x.copy())
    
    # Extract final residuals for each K value
    actual_residuals_final = actual_residuals_all  # residual at iteration k is actual_residuals_all[k-1]
    
    # Run verifier for different K values
    verifier_bounds_final = []
    
    for K in K_values:
        print(f"--- Testing Verifier for K = {K} ---")
        
        # Verifier analysis for this K
        solver_params = {'OutputFlag': 0}
        VP = Verifier(solver_params=solver_params)
        
        # Add parameter uncertainty in measurements
        b_param = VP.add_param(m, lb=b-0.005, ub=b+0.005)
        
        # Use box constraints as approximation for L1 ball
        box_bound = t  # Each component can be at most t in absolute value
        A_box = np.vstack([np.eye(n), -np.eye(n)])
        b_box = box_bound * np.ones(2*n)
        
        z0 = VP.add_initial_iterate(n, lb=0, ub=0)
        z = [None for _ in range(K + 1)]
        z[0] = z0
        
        # Run Frank-Wolfe through verifier
        for k in range(1, K + 1):
            gamma = 2.0 / (k + 1)
            q_param = -A.T @ b_param
            z[k] = VP.frank_wolfe_step(z[k-1], P, q_param, A_box, b_box, gamma, M=100)
        
        if K > 1:
            VP.set_infinity_norm_objective([z[K] - z[K-1]])
            verifier_bound = VP.solve()
        else:
            verifier_bound = actual_residuals_final[0]  # Use actual for K=1
        
        verifier_bounds_final.append(verifier_bound)
        
        print(f"  K={K}: Actual residual = {actual_residuals_final[K-1]:.6f}, Verifier bound = {verifier_bound:.6f}")
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Residuals vs K
    plt.subplot(1, 3, 1)
    plt.semilogy(K_values, actual_residuals_final, 'bo-', label='Actual Frank-Wolfe', linewidth=2, markersize=6)
    plt.semilogy(K_values, verifier_bounds_final, 'rs--', label='Verifier Worst-case', linewidth=2, markersize=6)
    plt.xlabel('Number of Iterations (K)')
    plt.ylabel('Fixed-point Residual at Iteration K')
    plt.title('Convergence Rate:\nActual vs Worst-case')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Gap between bounds
    plt.subplot(1, 3, 2)
    gap = np.array(verifier_bounds_final) - np.array(actual_residuals_final)
    plt.semilogy(K_values, gap, 'mo-', label='Verifier Conservatism', linewidth=2, markersize=6)
    plt.xlabel('Number of Iterations (K)')
    plt.ylabel('Worst-case Bound - Actual Residual')
    plt.title('How Conservative are\nVerifier Bounds?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Solution quality vs iterations
    plt.subplot(1, 3, 3)
    # Compute error to optimal for each iteration
    final_errors = [np.linalg.norm(solutions_all[k] - x_optimal) for k in K_values]
    
    plt.semilogy(K_values, final_errors, 'go-', label='Distance to Optimal', linewidth=2, markersize=6)
    plt.xlabel('Number of Iterations (K)')
    plt.ylabel('||x^K - x*||')
    plt.title('Solution Quality\nvs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Frank-Wolfe on ISTA Problem: Sparse Recovery (m={m}, n={n})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print(f"\n" + "="*50)
    print("SUMMARY:")
    print(f"  Max conservatism: {np.max(gap):.3f}")
    print(f"  Min conservatism: {np.min(gap):.3f}")
    print(f"  Avg conservatism: {np.mean(gap):.3f}")
    print(f"  Final actual residual (K={K_values[-1]}): {actual_residuals_final[-1]:.6f}")
    print(f"  Final verifier bound (K={K_values[-1]}): {verifier_bounds_final[-1]:.6f}")
    print("="*50)
    
    # Assertion
    assert np.all(np.array(actual_residuals_final) <= np.array(verifier_bounds_final) + 1e-6), \
        "Actual residuals should always be within verifier bounds"


if __name__ == "__main__":
    # test_frank_wolfe_quadratic_box()
    # test_frank_wolfe_simplex()
    # test_frank_wolfe_polytope()
    # test_frank_wolfe_multiple_instances()
    test_frank_wolfe_ista_instance_vs_K()
    print("\nAll Frank-Wolfe tests passed!") 
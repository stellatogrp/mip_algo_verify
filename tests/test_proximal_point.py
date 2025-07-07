import matplotlib.pyplot as plt
import numpy as np
from mipalgover.verifier import Verifier

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation

# Problem setup
n = 3  

np.random.seed(42)  # For reproducibility
M = np.random.randn(n, n)
P = M.T @ M + np.eye(n)
print("P matrix:")
print(P)

# Box constraints: 0 ≤ x ≤ 1
A = np.vstack([np.eye(n), -np.eye(n)])
b = np.hstack([np.ones(n), np.zeros(n)])  # x ≤ 1 and -x ≤ 0 (i.e. x ≥ 0)

K = 15  # Run until just before hitting time limit
lambd = 1.0  # Proximal parameter

def run_proximal_point(P, lambd):
    # Initialize verifier - turn off verbose output for cleaner results
    solver_params = {'OutputFlag': 0}  # No time limit
    VP = Verifier(solver_params=solver_params)
    
    q = VP.add_param(n, lb=-2, ub=3)
    
    z0 = VP.add_initial_iterate(n, lb=0.5, ub=0.5)  # Start in middle of feasible region
    z = [None for _ in range(K + 1)]
    z[0] = z0
    
    residuals = []
    for k in range(1, K+1):
        print(f"Iteration {k}/{K} (λ={lambd})")
        # z^{k+1} = argmin_v { f(v) + (1/2λ)||v - z^k||^2 : Av ≤ b }
        # where z^k is the previous iterate (reference point in proximal term)
        z[k] = VP.proximal_point_step(z[k-1], P, q, A, b, lambd, M=10)  # Smaller M
        VP.set_infinity_norm_objective([z[k] - z[k-1]])
        res = VP.solve()
        residuals.append(res)
        print(f"  Residual: {res}")
        
    print(f"\nResiduals for λ={lambd}, condition number {np.linalg.cond(P):.2f}:")
    print(residuals)
    return residuals

try:
    print("Running proximal point with λ=1.0...")
    residuals1 = run_proximal_point(P, lambd)
    
    # Test with different proximal parameter
    # lambd2 = 0.5
    # print(f"\nRunning proximal point with λ={lambd2}...")
    # residuals2 = run_proximal_point(P, lambd2)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, K+1), residuals1, 'b-o', label=f'λ={lambd}')
    #plt.plot(range(1, K+1), residuals2, 'r--s', label=f'λ={lambd2}')
    plt.yscale('log')
    plt.ylabel('Worst-case fixed-point residual')
    plt.xlabel(r'$K$')
    plt.title('Proximal Point Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Success! Proximal point implementation is working.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 
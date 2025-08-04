import matplotlib.pyplot as plt
# --- basic imports ---
import numpy as np
import gurobipy as gp
import time
from mipalgover.verifier import Verifier

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

n = 5             

rng = np.random.default_rng()
l_vec = rng.uniform(-2.0, -0.5, n)   # lower bounds  (all < 0)
u_vec = rng.uniform(0.5,  2.0,  n)   # upper bounds  (all > 0)

A = np.vstack([np.eye(n), -np.eye(n)])   
b = np.hstack([u_vec, -l_vec])           

Q = rng.standard_normal((n, n))
P = Q.T @ Q + 0.1 * np.eye(n)

x_lb, x_ub = -1.5, 1.5

K = 5

solver_params = {
    'OutputFlag': 0,
}
# VP = Verifier(solver_params=solver_params)
# x_param = VP.add_param(m, lb=x_lb, ub=x_ub)

# z_star = VP.add_initial_iterate(n, lb=-100, ub=100) 
# s_star = VP.linpro_step(P@z_star+x_param, A,b,1000,1000)

# VP.equality_constraint(z_star, s_star )


# for k in range(1, K+1):
#     s[k] = VP.linpro_step(P@z[k-1]+x_param, A,b,1000,1000)
#     alpha_k = 2/(k+2)
#     z[k] = VP.add_initial_iterate(n, lb=-100, ub=100)
#     VP.equality_constraint(z[k], (1-alpha_k)*z[k-1] + alpha_k*s[k])
#     #VP.set_infinity_norm_objective([z[k] - z[k-1]])
#     VP.set_infinity_norm_objective([z[k] - z_star])
#     res = VP.solve()
#     all_residuals.append(res)
#First run the verification
VP = Verifier(solver_params=solver_params)
x_param = VP.add_param(n, lb=x_lb, ub=x_ub)

z0 = VP.add_initial_iterate(n, lb=1, ub=1)
z = [None for _ in range(K + 1)]
z[0] = z0
s = [None for _ in range(K + 1)]
s0= VP.add_initial_iterate(n, lb=0.5, ub=0.5)
s[0]=s0
all_residuals = []  
start_vp = time.time()

# z_star = VP.add_initial_iterate(n, lb=-1, ub=3) 

# VP.equality_constraint(z_star, VP.linpro_step(P@z_star+x_param, A,b,100,100))

for k in range(1, K+1):
    s[k] = VP.linpro_step(P@z[k-1]+x_param, A,b,4,40,show_bounds=False)
    
    alpha_k = 2/(k+2)
    
    z[k] = VP.add_initial_iterate(n, lb=-10, ub=10)
    VP.equality_constraint(z[k], (1-alpha_k)*z[k-1] + alpha_k*s[k])
    
    VP.set_infinity_norm_objective([z[k] - z[k-1]])
    # VP.set_infinity_norm_objective([z[k] - z_star])
    res = VP.solve()
    all_residuals.append(res)
    print(f"VP iteration {k} done in {time.time() - start_vp:.1f}s")
vp_time = time.time() - start_vp

N = 10000 
sample_residuals = np.zeros((N, K))

for i in range(N):
    x_param_sample = np.random.uniform(x_lb, x_ub, n)
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    z_star_sample = model.addMVar(n, lb=-100)
    model.setObjective(1/2*z_star_sample.T@P@z_star_sample + x_param_sample.T@z_star_sample, gp.GRB.MINIMIZE)
    model.addConstr(A@z_star_sample <= b)
    model.optimize()
    z_star_sample = z_star_sample.X
    
    z_k = np.ones(n)
    
    for k in range(1, K+1):
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        s_k = model.addMVar(n, lb=-100)
        model.setObjective((P@z_k + x_param_sample)@s_k, gp.GRB.MINIMIZE)
        model.addConstr(A@s_k <= b)
        model.optimize()
        s_k = s_k.X
        
        alpha_k = 2/(k+2)
        z_k_new = (1-alpha_k)*z_k + alpha_k*s_k
        
        sample_residuals[i,k-1] = np.max(np.abs(z_k_new - z_k))
        
        z_k = z_k_new

plt.plot(range(1, K+1), all_residuals, label=f'VP ({vp_time:.1f}s)', color='blue')
plt.plot(range(1, K+1), np.max(sample_residuals, axis=0), label=f'Sample Max (N={N})', color='red', linestyle='--')
plt.yscale('log')
plt.ylabel('Worst-case fixed-point residual')
plt.xlabel(r'$K$')
# add a title
#plt.title(f'||z^k-z^k-1||_\infty VP vs Sample Max for {N} random samples')
plt.title(f'||z^k-z^k-1||_\infty VP vs Sample Max for {N} random samples with n={n} k={K}')
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------


import logging
import time

import cvxpy as cp
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as spa
from PEPit import PEP
from PEPit.functions import (
    ConvexIndicatorFunction,
    SmoothConvexLipschitzFunction,
)
from PEPit.primitive_steps import proximal_step

jnp.set_printoptions(precision=5)  # Print few decimal places
jnp.set_printoptions(suppress=True)  # Suppress scientific notation
jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


def test_with_cvxpy(cfg, A_supply, b_supply, A_demand, mu, z0):
    n = mu.shape[0]
    f = cp.Variable(n)
    obj = mu.T @ f

    b_demand = cfg.flow.x.demand_ub * jnp.ones(A_demand.shape[0])

    constraints = [
        A_supply @ f <= b_supply,
        A_demand @ f == b_demand,
        f >= 0,
        f <= cfg.flow.x.capacity_lb,
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)

    prob.solve()

    f_cp = f.value
    v_cp = constraints[0].dual_value
    w_cp = constraints[1].dual_value

    R = np.linalg.norm(f_cp - z0) ** 2 + np.linalg.norm(v_cp) ** 2 + np.linalg.norm(w_cp) ** 2

    log.info(f'R = {np.sqrt(R)}')


def LP_pep(cfg, A_supply, A_demand, mu, z0):

    K = cfg.K_max
    # K_min = cfg.K_min
    # momentum = cfg.momentum
    # m, n = A.shape
    # pnorm = cfg.pnorm
    m1 = A_supply.shape[0]
    m2 = A_demand.shape[0]

    log.info(z0)
    log.info(f'primal dim: {z0.shape}, supply_m: {m1}, supply_m: {m2}')
    Atilde = spa.vstack([A_supply, A_demand])
    M = spa.linalg.norm(Atilde, 2)
    t = cfg.rel_stepsize / spa.linalg.norm(Atilde, 2)
    log.info(M)
    log.info(f'using t = {t}')

    # b_supply = jnp.ones(A_supply.shape[0]) * cfg.flow.x.supply_lb  # TODO: change if parameterized

    # test_with_cvxpy(cfg, A_supply, b_supply, A_demand, mu, z0)

    # v0 = jnp.zeros(m1)
    # w0 = jnp.zeros(m2)

    R = 20.698  # use the above functions
    # t = .01

    taus = []
    solvetimes = []
    for k in range(1, K+1):
        if not cfg.momentum:
            tau, solvetime = vanilla_pep(k, R, M, t)
        else:
            tau, solvetime = momentum_pep(k, R, M, t)
        log.info(f'K={k}, tau={tau}')
        taus.append(tau)
        solvetimes.append(solvetime)

        log.info(taus)

        df = pd.DataFrame(taus)
        df.to_csv('taus.csv', index=False, header=False)

        df = pd.DataFrame(solvetimes)
        df.to_csv('times.csv', index=False, header=False)


def vanilla_pep(K, R, L, t, alpha=1, theta=1):
    problem = PEP()

    # func1 = problem.declare_function(ConvexFunction)
    func1 = problem.declare_function(SmoothConvexLipschitzFunction, L=L, M=L)
    # func2 = problem.declare_function(SmoothConvexFunction, L=L)
    # func2 = problem.declare_function(ConvexLipschitzFunction, M=L)
    func2 = problem.declare_function(ConvexIndicatorFunction)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(K)]
    w = [x0 for _ in range(K + 1)]
    for i in range(K):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= R ** 2)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    if K == 1:
        # problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
        problem.set_performance_metric((x[-1] - x0) ** 2 + (w[-1] - x0) ** 2)
    else:
        # problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)
        problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (w[-1] - w[-2]) ** 2)


    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='cvxpy')
    # pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return np.sqrt(pepit_tau), end - start


def momentum_pep(K, R, L, t, alpha=1, theta=1):
    problem = PEP()

    # TODO: see if there are fixes here

    # func1 = problem.declare_function(ConvexFunction)
    # func1 = problem.declare_function(ConvexLipschitzFunction, M=L)
    func1 = problem.declare_function(SmoothConvexLipschitzFunction, L=L, M=L)

    # func2 = problem.declare_function(SmoothConvexFunction, L=L)
    # func2 = problem.declare_function(ConvexLipschitzFunction, M=L)
    # func2 = problem.declare_function(SmoothConvexLipschitzFunction, L=L, M=L)
    func2 = problem.declare_function(ConvexIndicatorFunction)

    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2


    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(K)]
    w = [x0 for _ in range(K + 1)]
    u = [x0 for _ in range(K + 1)]

    for i in range(K):
        x[i], _, _ = proximal_step(u[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

        if i >= 1:
            u[i + 1] = w[i + 1] + (i - 1) / (i + 2) * (w[i + 1] - w[i])
        else:
            u[i + 1] = w[i + 1]

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 + (w[0] - xs) ** 2 <= R ** 2)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    if K == 1:
        # problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
        problem.set_performance_metric((x[-1] - x0) ** 2 + (w[-1] - x0) ** 2)
    else:
        # problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)
        problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (w[-1] - w[-2]) ** 2)


    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='cvxpy')
    # pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return np.sqrt(pepit_tau), end - start


def mincostflow_LP_run(cfg):
    log.info(cfg)
    flow = cfg.flow
    n_supply, n_demand, p, seed = flow.n_supply, flow.n_demand, flow.p, flow.seed

    G = nx.bipartite.random_graph(n_supply, n_demand, p, seed=seed, directed=False)
    A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)

    n_arcs = A.shape[1]
    A[n_supply:, :] *= -1

    log.info(A.todense())

    # t = cfg.rel_stepsize / spa.linalg.norm(A, ord=2)
    # log.info(f'using t={t}')

    key = jax.random.PRNGKey(flow.c.seed)
    c = jax.random.uniform(key, shape=(n_arcs,), minval=flow.c.low, maxval=flow.c.high)
    log.info(c)

    A_supply = A[:n_supply, :]
    A_demand = A[n_supply:, :]

    A_block = spa.bmat([
        [A_supply, spa.eye(n_supply), None],
        [A_demand, None, None],
        [spa.eye(n_arcs), None, spa.eye(n_arcs)]
    ])

    log.info(f'overall A size: {A_block.shape}')

    n_tilde = A_block.shape[1]
    c_tilde = np.zeros(n_tilde)
    c_tilde[:n_arcs] = c

    log.info(c_tilde)

    m, n = A_block.shape
    if flow.u0.type == 'high_demand':
        # x, _ = get_x_LB_UB(cfg, A_block)  # use the lower bound
        supply_lb = flow.x.supply_lb
        demand_lb = flow.x.demand_lb  # demand in our convention is negative so use the lower bound
        capacity_ub = flow.x.capacity_ub

        b_tilde = jnp.hstack([
            supply_lb * jnp.ones(flow.n_supply),
            demand_lb * jnp.ones(flow.n_demand),
            capacity_ub * jnp.ones(n_arcs),
        ])
        log.info(f'hardest x to satisfy: {b_tilde}')

        x_tilde = cp.Variable(n)

        constraints = [A_block @ x_tilde == b_tilde, x_tilde >= 0]

        prob = cp.Problem(cp.Minimize(c_tilde.T @ x_tilde), constraints)
        res = prob.solve(solver=cp.CLARABEL)
        log.info(res)

        if res == np.inf:
            log.info('the problem in the family with lowest supply and highest demand is infeasible')
            exit(0)

        u0 = x_tilde.value
        # v0 = constraints[0].dual_value
        v0 = jnp.zeros(m)

        log.info(f'u0: {u0}')
        log.info(f'v0: {v0}')

    else:
        if flow.u0.type == 'zero':
            u0 = jnp.zeros(n)
            # u0 = jnp.ones(n)  # TODO change back to zeros when done testing

        if flow.v0.type == 'zero':
            v0 = jnp.zeros(m)

    # LP_run(cfg, jnp.asarray(A_block.todense()), c_tilde, t, u0, v0)
    LP_pep(cfg, A_supply, A_demand, c, u0[:A_supply.shape[1]])


def pep(cfg):
    log.info(cfg)
    mincostflow_LP_run(cfg)

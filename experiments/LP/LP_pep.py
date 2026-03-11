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
    n = A_supply.shape[1]
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

    R = 19.4619  # use the above functions
    # t = .01
    t_inv = 1 / t

    taus = []
    solvetimes = []
    for k in range(cfg.K_min, K+1):
        if not cfg.momentum:
            H = spa.bmat([
                [t_inv * spa.eye(n), -A_supply.T, A_demand.T],
                [-A_supply, t_inv * spa.eye(m1), None],
                [-A_demand, None, t_inv * spa.eye(m2)]
            ])
            H_norm = spa.linalg.norm(H, 2)
            tau, solvetime = vanilla_pep(k, R, M, t, H_norm)
        else:
            xi = 1 + 2 * (k-1) / (k + 2)
            H = spa.bmat([
                [t_inv * spa.eye(n), -A_supply.T, A_demand.T],
                [-xi * A_supply, t_inv * spa.eye(m1), None],
                [-xi * A_demand, None, t_inv * spa.eye(m2)]
            ])
            H_norm = spa.linalg.norm(H, 2)
            tau, solvetime = momentum_pep(k, R, M, t, H_norm)
        log.info(f'K={k}, tau={tau}')
        taus.append(tau)
        solvetimes.append(solvetime)

        log.info(taus)

        df = pd.DataFrame(taus)
        df.to_csv('taus.csv', index=False, header=False)

        df = pd.DataFrame(solvetimes)
        df.to_csv('times.csv', index=False, header=False)


def vanilla_pep(K, R, M, t, H_norm, alpha=1, theta=1):
    problem = PEP()

    # f1 = problem.declare_function(ConvexFunction)
    f1 = problem.declare_function(SmoothConvexLipschitzFunction, L=M, M=M)
    h  = problem.declare_function(ConvexIndicatorFunction)
    func = f1 + h
    xs = func.stationary_point()

    # gs = func.gradient(xs)

    xs = problem.set_initial_point()
    ys = problem.set_initial_point()

    # Enforce these conditions in PEPit
    # f1.add_point(xs, g=-ys, f=f1.value(xs))
    # h.add_point(ys, g=xs,  f=h.value(ys))
    f1.add_point((xs, -M * ys, f1.value(xs)))
    h.add_point((ys, M * xs, h.value(ys)))

    # 3. Initialize the Algorithm
    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()

    # Constrain initial distance to the saddle point
    # We use the standard Euclidean norm for simplicity, though PDHG
    # is naturally contractive in the norm ||z||_M where M depends on tau/sigma.
    problem.set_initial_condition((x0 - xs)**2 + (y0 - ys)**2 <= R ** 2)

    x = x0
    y = y0

    for k in range(K):
        # --- Primal Step ---
        # x_{k+1} = prox_{tau f1}(x_k - tau * y_k)
        x_new, _, _ = proximal_step(x - t * M * y, f1, t)

        # --- Extrapolation ---
        # x_bar_{k+1} = x_{k+1} + theta * (x_{k+1} - x_k)
        x_bar = x_new + theta * (x_new - x)

        # --- Dual Step ---
        # y_{k+1} = prox_{sigma f2^*}(y_k + sigma * x_bar)
        # Note: prox_{sigma f2^*} is exactly prox_{sigma h}
        y_new, _, _ = proximal_step(y + t * M * x_bar, h, t)

        # Update
        y_prev = y
        x_prev = x

        x = x_new
        y = y_new

    # L_primal_view = f1.value(x) + M * (x * ys) - h.value(ys)
    # # Term 2: L(xs, y_avg) = f1(xs) + <xs, y_avg> - h(y_avg)
    # L_dual_view   = f1.value(xs) + M * (xs * y) - h.value(y)
    # gap = L_primal_view - L_dual_view
    # problem.set_performance_metric(gap)

    # obj = gs ** 2
    problem.set_performance_metric((x - x_prev) ** 2 + (y - y_prev) ** 2)

    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='cvxpy', solver='clarabel')
    # pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return H_norm * np.sqrt(pepit_tau), end - start


def momentum_pep(K, R, M, t, H_norm, alpha=1, theta=1):
    problem = PEP()

    # f1 = problem.declare_function(ConvexFunction)
    f1 = problem.declare_function(SmoothConvexLipschitzFunction, L=M, M=M)
    h  = problem.declare_function(ConvexIndicatorFunction)
    func = f1 + h
    xs = func.stationary_point()

    # gs = func.gradient(xs)

    xs = problem.set_initial_point()
    ys = problem.set_initial_point()

    # Enforce these conditions in PEPit
    # f1.add_point(xs, g=-ys, f=f1.value(xs))
    # h.add_point(ys, g=xs,  f=h.value(ys))
    f1.add_point((xs, -M * ys, f1.value(xs)))
    h.add_point((ys, M * xs, h.value(ys)))

    # 3. Initialize the Algorithm
    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()

    # Constrain initial distance to the saddle point
    # We use the standard Euclidean norm for simplicity, though PDHG
    # is naturally contractive in the norm ||z||_M where M depends on tau/sigma.
    problem.set_initial_condition((x0 - xs)**2 + (y0 - ys)**2 <= R ** 2)

    x = x0
    y = y0

    for k in range(K):
        # --- Primal Step ---
        # x_{k+1} = prox_{tau f1}(x_k - tau * y_k)
        x_new, _, _ = proximal_step(x - t * M * y, f1, t)

        if k >= 1:
            x_new = x_new + (k-1) / (k+2) * (x_new - x)

        # --- Extrapolation ---
        # x_bar_{k+1} = x_{k+1} + theta * (x_{k+1} - x_k)
        x_bar = x_new + theta * (x_new - x)

        # --- Dual Step ---
        # y_{k+1} = prox_{sigma f2^*}(y_k + sigma * x_bar)
        # Note: prox_{sigma f2^*} is exactly prox_{sigma h}
        y_new, _, _ = proximal_step(y + t * M * x_bar, h, t)

        # Update
        y_prev = y
        x_prev = x

        x = x_new
        y = y_new

    # L_primal_view = f1.value(x) + M * (x * ys) - h.value(ys)
    # # Term 2: L(xs, y_avg) = f1(xs) + <xs, y_avg> - h(y_avg)
    # L_dual_view   = f1.value(xs) + M * (xs * y) - h.value(y)
    # gap = L_primal_view - L_dual_view
    # problem.set_performance_metric(gap)

    # obj = gs ** 2
    problem.set_performance_metric((x - x_prev) ** 2 + (y - y_prev) ** 2)

    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='cvxpy', solver='clarabel')
    # pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return H_norm * np.sqrt(pepit_tau), end - start


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

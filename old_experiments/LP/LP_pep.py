import logging
import time

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as spa
from PEPit import PEP
from PEPit.functions import (
    ConvexFunction,
    ConvexLipschitzFunction,
)
from PEPit.primitive_steps import proximal_step
from tqdm import trange

jnp.set_printoptions(precision=5)  # Print few decimal places
jnp.set_printoptions(suppress=True)  # Suppress scientific notation
jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def get_x_LB_UB(cfg, A):
    # TODO: change if we decide to use something other than a box
    if cfg.problem_type == 'flow':
        # b_tilde = np.hstack([b_supply, b_demand, u])

        flow = cfg.flow
        flow_x = flow.x

        supply_lb, supply_ub = flow_x.supply_lb, flow_x.supply_ub
        demand_lb, demand_ub = flow_x.demand_lb, flow_x.demand_ub
        capacity_lb, capacity_ub = flow_x.capacity_lb, flow_x.capacity_ub

        log.info(A.shape)
        n_arcs = A.shape[0] - flow.n_supply - flow.n_demand
        log.info(n_arcs)

        lb = jnp.hstack([
            supply_lb * jnp.ones(flow.n_supply),
            demand_lb * jnp.ones(flow.n_demand),
            capacity_lb * jnp.ones(n_arcs),
        ])

        ub = jnp.hstack([
            supply_ub * jnp.ones(flow.n_supply),
            demand_ub * jnp.ones(flow.n_demand),
            capacity_ub * jnp.ones(n_arcs),
        ])

        log.info(lb)
        log.info(ub)
    else:
        m = A.shape[0]
        lb = cfg.x.l * jnp.ones(m)
        ub = cfg.x.u * jnp.ones(m)

    return lb, ub


def sample_radius(cfg, A, c, t, u0, v0):
    sample_idx = jnp.arange(cfg.samples.init_dist_N)
    m, n = A.shape

    # if cfg.u0.type == 'zero':
    #     u0 = jnp.zeros(n)

    # if cfg.v0.type == 'zero':
    #     v0 = jnp.zeros(m)

    def u_sample(i):
        # return jnp.zeros(n)
        return u0

    def v_sample(i):
        # return jnp.zeros(m)
        return v0

    # def x_sample(i):
    #     key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
    #     # TODO add the if, start with box case only
    #     return jax.random.uniform(key, shape=(cfg.m,), minval=cfg.x.l, maxval=cfg.x.u)

    x_LB, x_UB = get_x_LB_UB(cfg, A)
    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(m,), minval=x_LB, maxval=x_UB)

    u_samples = jax.vmap(u_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    distances = jnp.zeros(cfg.samples.init_dist_N)
    for i in trange(cfg.samples.init_dist_N):
        u = cp.Variable(n)
        x_samp = x_samples[i]
        obj = cp.Minimize(c @ u)

        constraints = [A @ u == x_samp, u >= 0]
        prob = cp.Problem(obj, constraints)

        prob.solve()
        # log.info(res)
        # distances.append(jnp.linalg.norm(z.value - z_samples[i]))
        # distances = distances.at[i].set(jnp.linalg.norm(z.value - z_samples[i]))
        u_val = u.value
        v_val = -constraints[0].dual_value
        z = np.hstack([u_val, v_val])
        z0 = np.hstack([u_samples[i], v_samples[i]])
        distances = distances.at[i].set(np.linalg.norm(z - z0))

    log.info(distances)

    return jnp.max(distances), u_val, v_val, x_samp


def init_dist(cfg, A, c, t, u0_val, v0_val):
    m, n = A.shape
    A = np.asarray(A)
    c = np.asarray(c)
    model = gp.Model()

    if cfg.x.type == 'box':
        # x_LB = cfg.x.l * np.ones(m)
        # x_UB = cfg.x.u * np.ones(m)
        x_LB, x_UB = get_x_LB_UB(cfg, A)

    # if cfg.u0.type == 'zero':
    #     u0_LB = np.zeros(n)
    #     u0_UB = np.zeros(n)

    # if cfg.v0.type == 'zero':
    #     v0_LB = np.zeros(m)
    #     v0_UB = np.zeros(m)

    u0_LB = u0_val
    u0_UB = u0_val

    v0_LB = v0_val
    v0_UB = v0_val

    bound_M = 110
    ustar = model.addMVar(n, lb=0, ub=bound_M)
    vstar = model.addMVar(m, lb=-bound_M, ub=bound_M)
    x = model.addMVar(m, lb=x_LB, ub=x_UB)
    u0 = model.addMVar(n, lb=u0_LB, ub=u0_UB)
    v0 = model.addMVar(m, lb=v0_LB, ub=v0_UB)
    w = model.addMVar(n, vtype=gp.GRB.BINARY)

    M = cfg.init_dist_M

    sample_rad, u_samp, v_samp, x_samp = sample_radius(cfg, A, c, t, u0_val, v0_val)
    log.info(f'sample radius: {sample_rad}')
    # exit(0)

    ustar.Start = u_samp
    vstar.Start = v_samp
    x.Start = x_samp

    # xkplus1 = np.maximum(xk - t * (c - A.T @ yk), 0)
    # ykplus1 = yk - t * (A @ (2 * xkplus1 - xk) - b)
    utilde = ustar - t * (c - A.T @ vstar)
    model.addConstr(ustar >= utilde)
    model.addConstr(vstar == vstar - t * (A @ ustar - x))
    for i in range(n):
        model.addConstr(ustar[i] <= utilde[i] + M * (1 - w[i]))
        model.addConstr(ustar[i] <= M * w[i])

    # TODO: incorporate the LP based bounding component wise
    model.addConstr(A @ ustar == x)
    model.addConstr(-A.T @ vstar + c >= 0)
    # model.addConstr(c @ ustar - x @ vstar == 0)

    z0 = gp.hstack([u0, v0])
    zstar = gp.hstack([ustar, vstar])

    # obj = (ustar - u0) @ (ustar - u0)

    obj = (zstar - z0) @ (zstar - z0)
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.optimize()

    log.info(f'sample radius: {sample_rad}')
    log.info(f'miqp max radius: {np.sqrt(model.objVal)}')
    return np.sqrt(model.objVal)


def pep(K, R, L, t, alpha=1, theta=1):
    problem = PEP()

    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(ConvexLipschitzFunction, M=L)

    # func = func1 + func2

    xs = func1.stationary_point()
    ys = func2.stationary_point()

    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()

    # zs = func.stationary_point()

    x = [x0 for _ in range(K)]
    w = [y0 for _ in range(K + 1)]
    y = [y0 for _ in range(K + 1)]

    for i in range(K):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y[i + 1], _, _ = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y[i + 1] - x[i])

    problem.set_initial_condition((x[0] - xs) ** 2 + (w[0] - ys) ** 2 <= R ** 2)

    if K == 1:
        # problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
        problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
    else:
        # problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)
        problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (w[-1] - w[-2]) ** 2)


    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return np.sqrt(pepit_tau), end - start


def momentum_pep(K, R, L, t, alpha=1, theta=1):
    problem = PEP()

    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(ConvexLipschitzFunction, M=L)

    xs = func1.stationary_point()
    ys = func2.stationary_point()

    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()

    x = [x0 for _ in range(K)]
    w = [y0 for _ in range(K + 1)]
    y = [y0 for _ in range(K + 1)]

    for i in range(K):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y[i + 1], _, _ = proximal_step(2 * x[i] - w[i], func1, alpha)
        beta_k = i/(i+3)
        if i == 1:
            w[i + 1] = w[i] + theta * (y[i + 1] - x[i])
        else:
            w[i + 1] = (1 + beta_k) * w[i] - beta_k * w[i-1] + beta_k * (y[i + 1] - x[i])

    problem.set_initial_condition((x[0] - xs) ** 2 + (w[0] - ys) ** 2 <= R ** 2)

    if K == 1:
        # problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
        problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
    else:
        # problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)
        problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (w[-1] - w[-2]) ** 2)
    # problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)


    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return np.sqrt(pepit_tau), end - start


def LP_pep(cfg, A, c, t, u0, v0):
    pep_rad = init_dist(cfg, A, c, t, u0, v0)
    L = np.linalg.norm(A, ord=2)

    K_max = cfg.K_max

    taus = []
    times = []
    for K in range(1, K_max + 1):
        log.info(f'----K={K}----')
        if cfg.momentum:
            tau, time = momentum_pep(K+1, pep_rad, L, t)
        else:
            tau, time = pep(K+1, pep_rad, L, t)
        taus.append(tau)
        times.append(time)

        df = pd.DataFrame(taus)
        df.to_csv(cfg.pep.resid_fname, index=False, header=False)

        df = pd.DataFrame(times)
        df.to_csv(cfg.pep.time_fname, index=False, header=False)

        log.info(taus)
        log.info(times)

    # log.info(taus)
    # log.info(times)


def random_LP_pep(cfg):
    log.info(cfg)
    m, n = cfg.m, cfg.n
    key = jax.random.PRNGKey(cfg.rng_seed)

    key, subkey = jax.random.split(key)
    A = jax.random.normal(subkey, shape=(m, n))

    key, subkey = jax.random.split(key)
    c = jax.random.uniform(subkey, shape=(n,))

    t = cfg.stepsize
    LP_pep(cfg, A, c, t)


def mincostflow_LP_pep(cfg):
    log.info(cfg)
    flow = cfg.flow
    n_supply, n_demand, p, seed = flow.n_supply, flow.n_demand, flow.p, flow.seed

    G = nx.bipartite.random_graph(n_supply, n_demand, p, seed=seed, directed=False)
    A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)

    n_arcs = A.shape[1]
    A[n_supply:, :] *= -1

    log.info(A.todense())

    t = cfg.rel_stepsize / spa.linalg.norm(A, ord=2)
    log.info(f'using t={t}')

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
        res = prob.solve()
        log.info(res)

        if res == np.inf:
            log.info('the problem in the family with lowest supply and highest demand is infeasible')
            exit(0)

        u0 = x_tilde.value
        v0 = constraints[0].dual_value

        log.info(f'u0: {u0}')
        log.info(f'v0: {v0}')

    else:
        if flow.u0.type == 'zero':
            u0 = jnp.zeros(n)
            # u0 = jnp.ones(n)  # TODO change back to zeros when done testing

        if flow.v0.type == 'zero':
            v0 = jnp.zeros(m)

    LP_pep(cfg, jnp.asarray(A_block.todense()), c_tilde, t, np.asarray(u0), np.asarray(v0))


def run(cfg):
    if cfg.problem_type == 'flow':
        mincostflow_LP_pep(cfg)
    else:
        random_LP_pep(cfg)

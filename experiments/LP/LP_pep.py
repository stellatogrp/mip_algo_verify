import logging
import time

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def sample_radius(cfg, A, c, t):
    sample_idx = jnp.arange(cfg.samples.init_dist_N)
    m, n = cfg.m, cfg.n

    # if cfg.u0.type == 'zero':
    #     u0 = jnp.zeros(n)

    # if cfg.v0.type == 'zero':
    #     v0 = jnp.zeros(m)

    def u_sample(i):
        return jnp.zeros(n)

    def v_sample(i):
        return jnp.zeros(m)

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.m,), minval=cfg.x.l, maxval=cfg.x.u)

    u_samples = jax.vmap(u_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    distances = jnp.zeros(cfg.samples.init_dist_N)
    for i in trange(cfg.samples.init_dist_N):
        u = cp.Variable(cfg.n)
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


def init_dist(cfg, A, c, t):
    m, n = cfg.m, cfg.n
    A = np.asarray(A)
    c = np.asarray(c)
    model = gp.Model()

    if cfg.x.type == 'box':
        x_LB = cfg.x.l * np.ones(m)
        x_UB = cfg.x.u * np.ones(m)

    if cfg.u0.type == 'zero':
        u0_LB = np.zeros(n)
        u0_UB = np.zeros(n)

    if cfg.v0.type == 'zero':
        v0_LB = np.zeros(m)
        v0_UB = np.zeros(m)

    bound_M = 110
    ustar = model.addMVar(n, lb=0, ub=bound_M)
    vstar = model.addMVar(m, lb=-bound_M, ub=bound_M)
    x = model.addMVar(m, lb=x_LB, ub=x_UB)
    u0 = model.addMVar(n, lb=u0_LB, ub=u0_UB)
    v0 = model.addMVar(m, lb=v0_LB, ub=v0_UB)
    w = model.addMVar(n, vtype=gp.GRB.BINARY)

    M = cfg.init_dist_M

    sample_rad, u_samp, v_samp, x_samp = sample_radius(cfg, A, c, t)
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

    # func1 = problem.declare_function(ConvexIndicatorFunction)
    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(ConvexLipschitzFunction, M=L)

    # func = func1 + func2

    xs = func1.stationary_point()
    ys = func2.stationary_point()

    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()

    # zs = func.stationary_point()

    x = [x0 for _ in range(K)]
    w = [x0 for _ in range(K + 1)]
    y = [y0 for _ in range(K + 1)]

    for i in range(K):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y[i + 1], _, _ = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y[i + 1] - x[i])

    problem.set_initial_condition((x[0] - xs) ** 2 + (y[0] - ys) ** 2 <= R ** 2)

    if K == 1:
        problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
    else:
        problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)
    # problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)


    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return np.sqrt(pepit_tau), end - start


def LP_pep(cfg, A, c, t):
    pep_rad = init_dist(cfg, A, c, t)
    L = np.linalg.norm(A, ord=2)
    log.info(L)

    K_max = cfg.K_max

    taus = []
    times = []
    for K in range(1, K_max + 1):
        log.info(f'----K={K}----')
        tau, time = pep(K, pep_rad, L, t)
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


def run(cfg):
    random_LP_pep(cfg)

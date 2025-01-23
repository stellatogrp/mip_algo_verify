import logging
import time

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as spa
from PEPit import PEP
from PEPit.functions import (
    ConvexIndicatorFunction,
    SmoothStronglyConvexQuadraticFunction,
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


def generate_P(cfg):
    n = cfg.n

    key = jax.random.PRNGKey(cfg.P_rng_seed)
    key, subkey = jax.random.split(key)
    # log.info(subkey)

    U = jax.random.orthogonal(subkey, n)
    out = jnp.zeros(n)

    key, subkey = jax.random.split(key)
    # log.info(subkey)
    out = out.at[1 : n - 1].set(
        jax.random.uniform(subkey, shape=(n - 2,), minval=cfg.mu, maxval=cfg.L)
    )

    out = out.at[0].set(cfg.mu)
    out = out.at[-1].set(cfg.L)

    if cfg.num_zero_eigvals > 0:
        out = out.at[1 : cfg.num_zero_eigvals + 1].set(0)

    P = U @ jnp.diag(out) @ U.T
    # eigs = jnp.linalg.eigvals(P)
    # log.info(f'eigval range: {jnp.min(eigs)} -> {jnp.max(eigs)}')
    # log.info(P)

    return P


def sample_radius(P, cfg):
    sample_idx = jnp.arange(cfg.samples.pep_sample_rad_N)

    def z_sample(i):
    # if cfg.z0.type == 'zero':
        # return jnp.zeros(n)
        return jnp.zeros(cfg.n)

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.n,), minval=cfg.x.l, maxval=cfg.x.u)

    z_samples = jax.vmap(z_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    log.info(x_samples)

    distances = jnp.zeros(cfg.samples.pep_sample_rad_N)
    for i in trange(cfg.samples.pep_sample_rad_N):
        z = cp.Variable(cfg.n)
        x_samp = x_samples[i]
        obj = cp.Minimize(.5 * cp.quad_form(z, P) + x_samp @ z)
        prob = cp.Problem(obj, [z >= 0])
        prob.solve()
        # distances.append(jnp.linalg.norm(z.value - z_samples[i]))
        distances = distances.at[i].set(jnp.linalg.norm(z.value - z_samples[i]))

    log.info(distances)

    return jnp.max(distances)


def pep_radius(cfg):
    P = generate_P(cfg)

    if cfg.stepsize.type == 'rel':
        t = cfg.stepsize.h / cfg.L
    elif cfg.stepsize.type == 'opt':
        t = 2 / (cfg.mu + cfg. L)
    elif cfg.stepsize.type == 'abs':
        t = cfg.stepsize.h

    # A = spa.eye(cfg.n) - t * P
    A = np.asarray(np.eye(cfg.n) - t * P)
    B = -t * spa.eye(cfg.n)
    # K_max = cfg.K_max
    # K_min = cfg.K_min

    sample_rad = sample_radius(P, cfg)
    log.info(f'sample radius: {sample_rad}')

    n = cfg.n
    model = gp.Model()

    if cfg.x.type == 'box':
        x_LB = cfg.x.l * jnp.ones(n)
        x_UB = cfg.x.u * jnp.ones(n)

    if cfg.z0.type == 'zero':
        z0_LB = jnp.zeros(n)
        z0_UB = jnp.zeros(n)

    zstar = model.addMVar(n, lb=0, ub=np.inf)  # works b/c of constraints in NNQP
    x = model.addMVar(n, lb=x_LB, ub=x_UB)
    z0 = model.addMVar(n, lb=z0_LB, ub=z0_UB)
    w = model.addMVar(n, vtype=gp.GRB.BINARY)

    M = 100  # put this in through cfg

    ystar = A @ zstar + B @ x
    model.addConstr(zstar >= ystar)
    for i in range(n):
        model.addConstr(zstar[i] <= ystar[i] + M * (1 - w[i]))
        model.addConstr(zstar[i] <= M * w[i])

    obj = (zstar - z0) @ (zstar - z0)
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.optimize()

    log.info(f'difference in miqp vs sm: {np.sqrt(model.objVal) - sample_rad}')
    log.info(x.X)
    log.info(zstar.X)

    return np.sqrt(model.objVal)


def single_pep_sample(t, mu, L, r, K, cfg):
    verbose=2
    problem = PEP()
    print(mu, L)
    # L = 74.659
    # mu = .1
    # t = 2 / (L + mu)
    # r = 1
    # K = 2

    # Declare a convex and a smooth convex function.
    func1 = problem.declare_function(ConvexIndicatorFunction)
    # func1 = problem.declare_function(ConvexFunction)
    # func2 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    func2 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # xs = func2.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(K + 1)]
    # w = [x0 for _ in range(N + 1)]
    # x = x0
    for i in range(K):
        y = x[i] - t * func2.gradient(x[i])
        x[i+1], _, _ = proximal_step(y, func1, t)

        # x[i + 1] = x[i] - t * func2.gradient(x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)
    # problem.set_initial_condition((x[1] - x[0]) ** 2 <= r ** 2)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    problem.set_performance_metric(cfg.obj_scaling * (x[-1] - x[-2]) ** 2)

    # Solve the PEP

    start = time.time()
    pepit_verbose = max(verbose, 0)
    try:
        pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
    except AssertionError:
        pepit_tau = problem.objective.eval()
    end = time.time()

    return np.sqrt(pepit_tau / cfg.obj_scaling), end-start


def NNQP_run(cfg):
    log.info(cfg)

    if cfg.stepsize.type == 'rel':
        t = cfg.stepsize.h / cfg.L
    elif cfg.stepsize.type == 'opt':
        t = 2 / (cfg.mu + cfg. L)
    elif cfg.stepsize.type == 'abs':
        t = cfg.stepsize.h

    R = pep_radius(cfg)
    log.info(R)

    pep_taus = []
    solvetimes = []

    for k in range(1, cfg.K_max):
        log.info(f'---solving PEP with k={k}----')
        tau, solvetime = single_pep_sample(t, cfg.mu, cfg.L, R, k, cfg)
        log.info(f'tau={tau}')
        log.info(f'time: {solvetime:.3f} s')

        pep_taus.append(tau)
        solvetimes.append(solvetime)

        df = pd.DataFrame(pep_taus)  # remove the first column of zeros
        df.to_csv(cfg.pep_resid_fname, index=False, header=False)

        df_times = pd.DataFrame(solvetimes)
        df_times.to_csv(cfg.pep_solvetime_fname, index=False, header=False)

        fig, ax = plt.subplots()
        ax.plot(range(1, len(pep_taus)+1), pep_taus, label='PEP Taus')

        ax.set_xlabel(r'$K$')
        ax.set_ylabel('Fixed-point residual')
        ax.set_yscale('log')
        ax.set_title(r'NNQP VP')

        ax.legend()

        plt.savefig(cfg.pep_resid_plot_fname)

        plt.clf()
        plt.cla()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(range(1, len(solvetimes)+1), solvetimes, label='PEP')

        ax.set_xlabel(r'$K$')
        ax.set_ylabel('Solvetimes (s)')
        ax.set_yscale('log')
        ax.set_title(r'NNQP VP')

        ax.legend()

        plt.savefig(cfg.pep_solvetime_plot_fname)

        plt.clf()
        plt.cla()
        plt.close()

    log.info(f'taus: {pep_taus}')
    log.info(f'solvetimes: {solvetimes}')


def run(cfg):
    NNQP_run(cfg)

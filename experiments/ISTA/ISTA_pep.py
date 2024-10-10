import logging
import time

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PEPit import PEP
from PEPit.functions import (
    ConvexLipschitzFunction,
    SmoothStronglyConvexFunction,
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


def sample_rad(cfg, A, c_z):
    sample_idx = jnp.arange(cfg.samples.init_dist_N)
    m, n = cfg.m, cfg.n

    def z_sample(i):
        return c_z

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(m,), minval=cfg.x.l, maxval=cfg.x.u)

    z_samples = jax.vmap(z_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)
    distances = jnp.zeros(cfg.samples.init_dist_N)

    for i in trange(cfg.samples.init_dist_N):
        z = cp.Variable(n)
        x = x_samples[i]
        obj = cp.Minimize(.5 * cp.sum_squares(A @ z - x) + cp.norm(z, 1))
        prob = cp.Problem(obj)
        prob.solve()

        z0 = z_samples[i]
        distances = distances.at[i].set(np.linalg.norm(z.value - z0))

    return jnp.max(distances)


def pep(K, R, mu, L, t, lambd, verbose=1):
    problem = PEP()
    f = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    # f = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    h = problem.declare_function(ConvexLipschitzFunction, M=1)
    F = f + h

    zs = F.stationary_point()

    z0 = problem.set_initial_point()

    problem.set_initial_condition((z0 - zs) ** 2 <= R ** 2)

    z = [z0 for _ in range(K+1)]
    for i in range(K):
        z[i + 1], _, _ = proximal_step(z[i] - t * f.gradient(z[i]), h, t * lambd)

    problem.set_performance_metric((z[-1] - z[-2]) ** 2)

    # tau = problem.solve(verbose=1, wrapper='mosek')
    pepit_verbose = max(verbose, 0)

    start = time.time()
    try:
        pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
    except AssertionError:
        pepit_tau = problem.objective.eval()
    end = time.time()

    return np.sqrt(pepit_tau), end - start


def generate_data(cfg):
    m, n = cfg.m, cfg.n
    k = min(m, n)
    mu, L = cfg.mu, cfg.L

    key = jax.random.PRNGKey(cfg.A_rng_seed)

    key, subkey = jax.random.split(key)
    sigma = jnp.zeros(k)
    sigma = sigma.at[1:k-1].set(jax.random.uniform(subkey, shape=(k-2,), minval=jnp.sqrt(mu), maxval=jnp.sqrt(L)))
    sigma = sigma.at[0].set(jnp.sqrt(mu))
    sigma = sigma.at[-1].set(jnp.sqrt(L))
    # log.info(sigma)

    key, subkey = jax.random.split(key)
    U = jax.random.orthogonal(subkey, m)

    key, subkey = jax.random.split(key)
    VT = jax.random.orthogonal(subkey, n)

    diag_sigma = jnp.zeros((m, n))
    diag_sigma = diag_sigma.at[jnp.arange(k), jnp.arange(k)].set(sigma)

    return U @ diag_sigma @ VT


def lstsq_sol(cfg, A):
    m, n = cfg.m, cfg.n
    lambd = cfg.lambd

    if cfg.x.type == 'box':
        x_l = cfg.x.l
        x_u = cfg.x.u

    key = jax.random.PRNGKey(cfg.x.seed)
    x_samp = jax.random.uniform(key, shape=(m,), minval=x_l, maxval=x_u)
    # log.info(x_samp)

    x_lstsq, _, _, _ = jnp.linalg.lstsq(A, x_samp)
    log.info(f'least squares sol: {x_lstsq}')

    z = cp.Variable(n)

    obj = cp.Minimize(.5 * cp.sum_squares(A @ z - x_samp) + lambd * cp.norm(z, 1))
    prob = cp.Problem(obj)
    prob.solve()

    log.info(f'lasso sol with lambda={lambd}: {z.value}')

    return x_lstsq


def ISTA_pep(cfg):
    log.info(cfg)

    m, n = cfg.m, cfg.n
    L = cfg.L
    if m < n:
        mu = 0
    else:
        mu = cfg.mu
    log.info(cfg)

    A = generate_data(cfg)
    A_eigs = jnp.real(jnp.linalg.eigvals(A.T @ A))
    log.info(f'eigenvalues of ATA: {A_eigs}')

    z_lstsq = lstsq_sol(cfg, A)

    if cfg.z0.type == 'lstsq':
        c_z = z_lstsq
    elif cfg.z0.type == 'zero':
        c_z = jnp.zeros(n)

    lambda_t = cfg.lambd * cfg.t
    log.info(f'lambda * t: {lambda_t}')

    pep_rad = float(sample_rad(cfg, A, c_z))

    log.info(pep_rad)

    K_max = cfg.K_max

    taus = []
    times = []
    for K in range(1, K_max + 1):
        log.info(f'----K={K}----')
        tau, time = pep(K, pep_rad, mu, L, cfg.t, cfg.lambd)
        taus.append(tau)
        times.append(time)

        df = pd.DataFrame(taus)
        df.to_csv(cfg.pep.resid_fname, index=False, header=False)

        df = pd.DataFrame(times)
        df.to_csv(cfg.pep.time_fname, index=False, header=False)

        log.info(taus)
        log.info(times)

    log.info(taus)
    log.info(times)


def run(cfg):
    ISTA_pep(cfg)

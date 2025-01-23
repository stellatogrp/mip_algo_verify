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
    # f = problem.declare_function(SmoothConvexFunction, L=L)
    f = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    # # f = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    h = problem.declare_function(ConvexLipschitzFunction, M=1)
    F = f + h

    zs = F.stationary_point()

    z0 = problem.set_initial_point()

    problem.set_initial_condition((z0 - zs) ** 2 <= R ** 2)

    z = [z0 for _ in range(K+1)]
    lambd_t = t * lambd
    lambd_t = float(lambd_t)
    t = float(t)
    for i in range(K):
        # yi = z[i] - t * f.gradient(z[i])
        # z[i + 1], _, _ = proximal_step(yi, h, lambd)
        y = z[i] - t * f.gradient(z[i])
        z[i + 1], _, _ = proximal_step(y, h, lambd_t)

    problem.set_performance_metric((z[-1] - z[-2]) ** 2)

    # tau = problem.solve(verbose=1, wrapper='mosek')
    pepit_verbose = max(verbose, 0)

    start = time.time()
    # try:
    #     pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
    # except AssertionError:
    #     pepit_tau = problem.objective.eval()
    pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
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


def lstsq_sol(cfg, A, lambd, x_l, x_u):
    m, n = cfg.m, cfg.n

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


def sparse_coding_A(cfg):
    m, n = cfg.m, cfg.n
    key = jax.random.PRNGKey(cfg.A_rng_seed)

    key, subkey = jax.random.split(key)
    A = 1 / m * jax.random.normal(subkey, shape=(m, n))

    # A_mask = jax.random.bernoulli(key, p=cfg.x_star.A_mask_prob, shape=(m-1, n)).astype(jnp.float64)

    # masked_A = jnp.multiply(A[1:], A_mask)

    # A = A.at[1:].set(masked_A)
    # return A / jnp.linalg.norm(A, axis=0)
    A_mask = jax.random.bernoulli(key, p=cfg.x_star.A_mask_prob, shape=(m, n)).astype(jnp.float64)
    masked_A = jnp.multiply(A, A_mask)
    # log.info(masked_A)

    for i in range(n):
        Ai = masked_A[:, i]
        if jnp.linalg.norm(Ai) > 0:
            masked_A = masked_A.at[:, i].set(Ai / jnp.linalg.norm(Ai))

    # log.info(jnp.linalg.norm(masked_A, axis=0))
    # log.info(jnp.count_nonzero(masked_A.T @ masked_A))
    # exit(0)

    return masked_A


def sparse_coding_b_set(cfg, A):
    m, n = A.shape

    key = jax.random.PRNGKey(cfg.x_star.rng_seed)

    key, subkey = jax.random.split(key)
    x_star_set = cfg.x_star.std * jax.random.normal(subkey, shape=(n, cfg.x_star.num))

    key, subkey = jax.random.split(key)
    x_star_mask = jax.random.bernoulli(subkey, p=cfg.x_star.nonzero_prob, shape=(n, cfg.x_star.num))

    x_star = jnp.multiply(x_star_set, x_star_mask)
    # log.info(x_star)

    epsilon = cfg.x_star.epsilon_std * jax.random.normal(key, shape=(m, cfg.x_star.num))

    b_set = A @ x_star + epsilon

    # log.info(A @ x_star)
    # log.info(b_set)

    return b_set


def sparse_coding_ISTA_pep(cfg):
    # m, n = cfg.m, cfg.n
    # n = cfg.n
    log.info(cfg)

    A = sparse_coding_A(cfg)
    m, n = A.shape

    log.info(A)

    A_eigs = jnp.real(jnp.linalg.eigvals(A.T @ A))
    log.info(f'eigenvalues of ATA: {A_eigs}')

    L = jnp.max(A_eigs)

    # log.info(A)
    # log.info(jnp.linalg.norm(A, axis=0))

    # x_star_set = sparse_coding_x_star(cfg, A)
    b_set = sparse_coding_b_set(cfg, A)

    x_l = jnp.min(b_set, axis=1)
    x_u = jnp.max(b_set, axis=1)

    log.info(f'size of x set: {x_u - x_l}')

    t = cfg.t_rel / L

    if cfg.lambd.val == 'adaptive':
        center = x_u - x_l
        lambd = cfg.lambd.scalar * jnp.max(jnp.abs(A.T @ center))
    else:
        lambd = cfg.lambd.val

    log.info(f't={t}')
    log.info(f'lambda = {lambd}')
    log.info(f'lambda * t = {lambd * t}')

    if cfg.z0.type == 'lstsq':
        c_z = lstsq_sol(cfg, A, lambd, x_l, x_u)
    elif cfg.z0.type == 'zero':
        c_z = jnp.zeros(n)

    A_eigs = jnp.real(jnp.linalg.eigvals(A.T @ A))
    log.info(f'eigenvalues of ATA: {A_eigs}')

    # L = jnp.max(A_eigs)
    if m < n:
        mu = 0
    else:
        mu = jnp.min(A_eigs)
    log.info(L)
    log.info(mu)

    pep_rad = float(sample_rad(cfg, A, c_z))

    log.info(pep_rad)

    # ISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u)

    taus = []
    times = []
    for K in range(1, cfg.K_max + 1):
        log.info(f'----K={K}----')
        tau, time = pep(K, pep_rad, float(mu), float(L), t, lambd)
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
    # ISTA_pep(cfg)
    if cfg.problem_type == 'random':
        # random_ISTA_run(cfg)
        ISTA_pep(cfg)
    elif cfg.problem_type == 'sparse_coding':
        sparse_coding_ISTA_pep(cfg)

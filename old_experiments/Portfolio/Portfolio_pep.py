# l2 R: 2.157

import logging
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PEPit import PEP
from PEPit.functions import (
    ConvexFunction,
    SmoothStronglyConvexFunction,
)
from PEPit.primitive_steps import proximal_step

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


def pep(K, R, mu, L, alpha=1, theta=1):
    problem = PEP()

    func1 = problem.declare_function(ConvexFunction)
    # func2 = problem.declare_function(ConvexLipschitzFunction, M=L)
    func2 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

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
    # problem.set_initial_condition((w[0] - ys) ** 2 <= R ** 2 )

    # if K == 1:
    #     # problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
    #     problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
    # else:
    #     problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)
    problem.set_performance_metric((w[-1] - w[-2]) ** 2)

    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return np.sqrt(pepit_tau), end - start


def portfolio_l2(cfg):
    log.info(cfg)
    n, d = cfg.n, cfg.d

    key = jax.random.PRNGKey(cfg.data_rng_key)

    key, subkey = jax.random.split(key)
    F = jax.random.normal(subkey, shape=(n, d))

    key, subkey = jax.random.split(key)
    F_mask = jax.random.bernoulli(subkey, p=cfg.F_mask_prob, shape=(n, d)).astype(jnp.float64)

    F = jnp.multiply(F, F_mask)
    log.info(F)

    key, subkey = jax.random.split(key)
    Ddiag = jax.random.uniform(subkey, shape=(n, ), maxval = 1/jnp.sqrt(d))

    D = jnp.diag(Ddiag)
    log.info(D)

    # if cfg.zprev.incl_upper_bound:
    #     A = np.block([
    #         [F.T, -jnp.eye(d)],
    #         [jnp.ones((1, n)), jnp.zeros((1, d))],
    #         [-jnp.eye(n), jnp.zeros((n, d))],
    #         [jnp.eye(n), jnp.zeros((n, d))]
    #     ])
    #     b = jnp.hstack([jnp.zeros(d), 1, jnp.zeros(n), cfg.zprev.u * jnp.ones(n)])
    # else:
    #     A = np.block([
    #         [F.T, -jnp.eye(d)],
    #         [jnp.ones((1, n)), jnp.zeros((1, d))],
    #         [-jnp.eye(n), jnp.zeros((n, d))]
    #     ])
    #     b = jnp.hstack([jnp.zeros(d), 1, jnp.zeros(n)])

    # log.info(A.shape)
    # log.info(b.shape)

    # mu_l = cfg.mu.l * jnp.ones(n)
    # mu_u = cfg.mu.u * jnp.ones(n)

    # if cfg.z0.type == 'avg_sol':
    #     key, subkey = jax.random.split(key)
    #     mu_sample = jax.random.uniform(subkey, shape=(n,), minval=mu_l, maxval=mu_u)
    #     s0 = avg_sol(cfg, D, A, b, mu_sample)
    # elif cfg.z0.type == 'zero':
    #     s0 = jnp.zeros(2 * n + 2 * d + 1)

    # y0 = F.T @ z0

    # portfolio_verifier(cfg, D, A, b, jnp.hstack([z0, y0]), mu_l, mu_u)
    # portfolio_verifier(cfg, D, A, b, s0, mu_l, mu_u)

    gamma = cfg.gamma
    lambd = cfg.lambd
    num_stocks, num_factors = cfg.n, cfg.d

    P = 2 * jnp.block([
        [gamma * D + lambd * jnp.eye(num_stocks), jnp.zeros((num_stocks, num_factors))],
        [jnp.zeros((num_factors, num_stocks)), gamma * jnp.eye(num_factors)]
    ])

    eigvals = jnp.real(jnp.linalg.eigvals(P))
    log.info(eigvals)

    # mu = jnp.min(eigvals)
    L = jnp.max(eigvals)
    R = 2.157

    # tau, solvetime = pep(2, R, float(mu), float(L))
    # log.info(tau)

    taus = []
    solvetimes = []

    for k in range(1, 31):
        tau, solvetime = pep(k, R, 0, float(L))

        taus.append(tau)
        solvetimes.append(solvetime)
        log.info(f'k = {k}, tau={tau}')

        df = pd.DataFrame(taus)
        df.to_csv(cfg.pep.resid_fname, index=False, header=False)

        df = pd.DataFrame(solvetimes)
        df.to_csv(cfg.pep.time_fname, index=False, header=False)

    log.info(taus)
    log.info(solvetimes)

def run(cfg):
    portfolio_l2(cfg)

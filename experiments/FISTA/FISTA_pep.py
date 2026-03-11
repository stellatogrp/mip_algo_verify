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


def solve_pep(K, R, mu, L, t, lambd, H_norm, verbose=1):
    K_max = K
    beta = jnp.ones(K_max + 1)
    gamma = jnp.zeros(K_max + 1)
    for k in range(1, K_max+1):
        beta = beta.at[k].set(.5 * (1 + jnp.sqrt(1 + 4 * jnp.power(beta[k-1], 2))))
        gamma = gamma.at[k].set((beta[k-1] - 1) / beta[k])

    gamma = np.asarray(gamma)

    problem = PEP()
    f = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    # f = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    h = problem.declare_function(ConvexLipschitzFunction, M=lambd)
    F = f + h

    zs = F.stationary_point()

    z0 = problem.set_initial_point()

    problem.set_initial_condition((z0 - zs) ** 2 <= R ** 2)

    z = [z0 for _ in range(K+1)]
    w = [z0 for _ in range(K+1)]
    lambd_t = t * lambd
    lambd_t = float(lambd_t)
    t = float(t)
    for i in range(K):
        y = w[i] - t * f.gradient(w[i])
        z[i + 1], _, _ = proximal_step(y, h, t)
        w[i + 1] = z[i + 1] + gamma[i+1] * (z[i + 1] - z[i])


    problem.set_performance_metric((z[-1] - z[-2]) ** 2)

    pepit_verbose = max(verbose, 0)

    start = time.time()
    # try:
    #     pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
    # except AssertionError:
    #     pepit_tau = problem.objective.eval()
    pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='cvxpy', solver='clarabel')
    # pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
    end = time.time()

    return H_norm * np.sqrt(pepit_tau), end - start


def nonstrong_cvx_pep(cfg):
    R = 34.608072156
    L = 3.70248
    mu = 0.
    lambd = 0.05
    t = 0.1350447081806767
    H_norm = 7.40495509577

    K_max = 40

    taus = []
    times = []

    for K in range(1, K_max+1):
        tau, solvetime = solve_pep(K, R, mu, L, t, lambd, H_norm)
        log.info(f'K={K}, tau = {tau}')

        taus.append(tau)
        times.append(solvetime)

        df = pd.DataFrame(taus)
        df.to_csv('taus.csv', index=False, header=False)

        df = pd.DataFrame(times)
        df.to_csv('times.csv', index=False, header=False)


def strong_cvx_pep(cfg):
    R = 50.25558419254079
    L = 2.82747
    mu = 0.02123
    lambd = 0.05
    t = 0.17683622357271744
    H_norm = 5.654949985904822

    K_max = 40

    taus = []
    times = []

    for K in range(1, K_max+1):
        tau, solvetime = solve_pep(K, R, mu, L, t, lambd, H_norm)
        log.info(f'K={K}, tau = {tau}')

        taus.append(tau)
        times.append(solvetime)

        df = pd.DataFrame(taus)
        df.to_csv('taus.csv', index=False, header=False)

        df = pd.DataFrame(times)
        df.to_csv('times.csv', index=False, header=False)


def pep(cfg):
    log.info(cfg)

    if cfg.m < cfg.n:
        log.info('nonstrong cvx')
        nonstrong_cvx_pep(cfg)

    if cfg.m > cfg.n:
        log.info('strong cvx')
        strong_cvx_pep(cfg)

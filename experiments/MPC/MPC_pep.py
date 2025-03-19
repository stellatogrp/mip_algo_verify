import logging
import time

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as spa
from MPC.quadcopter_compact import Quadcopter
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


def single_pep_run(K, r, mu, L):
    problem = PEP()
    alpha = 1
    theta = 1

    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    # func2 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    # func2 = problem.declare_function(ConvexFunction)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    x = [x0 for _ in range(K)]
    w = [x0 for _ in range(K + 1)]
    for i in range(K):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= r ** 2)
    problem.set_initial_condition((w[0] - xs) ** 2 <= r ** 2)

    if K == 1:
        # problem.set_performance_metric((x[-1] - x0) ** 2 + (y[-1] - y0) ** 2)
        problem.set_performance_metric((x[-1] - x0) ** 2 + (w[-1] - x0) ** 2)
    else:
        # problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)
        problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (w[-1] - w[-2]) ** 2)

    start = time.time()
    pepit_tau = problem.solve(verbose=1, wrapper='cvxpy', solver=cp.CLARABEL)
    # pepit_tau = problem.solve(verbose=1, wrapper='mosek')
    end = time.time()
    return np.sqrt(pepit_tau), end - start


def osqp_pep(cfg, qc, P, q, A, l, u):
    K_max = 50
    R = 9.001

    P_eigs, _ = spa.linalg.eigs(P)
    P_eigs = np.real(P_eigs)
    log.info(P_eigs)

    mu = np.min(P_eigs)
    L = np.max(P_eigs)

    taus = []
    solvetimes = []
    for k in range(40, K_max+1):
        tau, solvetime = single_pep_run(k, R, 1, L / mu)
        log.info(f'K={k}, tau={tau}')
        taus.append(tau)
        solvetimes.append(solvetime)

        log.info(taus)

        df = pd.DataFrame(taus)
        df.to_csv('taus.csv', index=False, header=False)

        df = pd.DataFrame(solvetimes)
        df.to_csv('times.csv', index=False, header=False)


def pep(cfg):
    log.info(cfg)
    qc = Quadcopter(T=cfg.T)
    # P, q, A, l, u = qc.P, qc.q, qc.A, qc.l, qc.u
    P, q, A, l, u, _ = qc.test_simplified_cvxpy()

    osqp_pep(cfg, qc, P, q, A, l, u)

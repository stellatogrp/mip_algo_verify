import logging

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mipalgover.verifier import Verifier

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


def ISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u):
    # pnorm = cfg.pnorm
    m, n = cfg.m, cfg.n
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T

    At = np.asarray(At)
    Bt = np.asarray(Bt)
    lambda_t = lambd * t

    K_max = cfg.K_max

    max_sample_resids = samples(cfg, A, lambd, t, c_z, x_l, x_u)
    log.info(max_sample_resids)

    gurobi_params = {
        'TimeLimit': cfg.timelimit,
        'MIPGap': cfg.mipgap,
    }

    def theory_func(k):
        if k == 1:
            return np.inf
        # return 2 * init_C / np.sqrt((k-1) * (k+2))
        return np.inf

    VP = Verifier(solver_params=gurobi_params, theory_func=theory_func)

    x_param = VP.add_param(m, lb=np.array(x_l), ub=np.array(x_u))
    z0 = VP.add_initial_iterate(n, lb=np.array(c_z), ub=np.array(c_z))

    z = [None for _ in range(K_max+1)]
    z[0] = z0

    Deltas = []
    rel_LP_sols = []
    Delta_bounds = []
    Delta_gaps = []
    times = []
    # theory_improv_fracs = []

    for k in range(1, K_max+1):
        log.info(f'Solving VP at k={k}')

        z[k] = VP.soft_threshold_step(At @ z[k-1] + Bt @ x_param, lambda_t, relax_binary_vars=(k >= cfg.relax_cutoff))
        # theory_improv = VP.theory_bound(k, z[k], z[k-1])

        VP.set_infinity_norm_objective(z[k] - z[k-1])
        VP.solve(huchette_cuts=cfg.huchette_cuts, include_rel_LP_sol=True)

        data = VP.extract_solver_data()
        print(data)

        Deltas.append(data['objVal'])
        rel_LP_sols.append(data['rel_LP_sol'])
        Delta_bounds.append(data['objBound'])
        Delta_gaps.append(data['MIPGap'])
        times.append(data['Runtime'])
        # theory_improv_fracs.append(theory_improv)

        plot_data(cfg, n, m, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, times)

        print(f'samples: {max_sample_resids}')
        print(f'rel LP sols: {jnp.array(rel_LP_sols)}')
        print(f'VP residuals: {jnp.array(Deltas)}')
        print(f'VP residual bounds: {jnp.array(Delta_bounds)}')
        print(f'times:{jnp.array(times)}')
        # print(f'theory improv fracs: {jnp.array(theory_improv_fracs)}')

    # x = VP.extract_sol(x_param)
    # log.info(f'testing at K={cfg.K_max}')
    # ISTA_true, true_resids = ISTA_alg(At, Bt, c_z, jnp.array(x), lambda_t, cfg.K_max, pnorm=cfg.pnorm)
    # log.info(ISTA_true)

    # for k in range(0, cfg.K_max + 1):
    #     ztest = VP.extract_sol(z[k])
    #     log.info(f'K={k}')
    #     log.info(ztest)

    # log.info(f'with x={x}')
    # log.info(f'x_l = {x_l}')
    # log.info(f'x_u = {x_u}')


def plot_data(cfg, n, m, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, solvetimes):
    df = pd.DataFrame(Deltas)  # remove the first column of zeros
    df.to_csv('resids.csv', index=False, header=False)

    df = pd.DataFrame(Delta_bounds)
    df.to_csv('resid_bounds.csv', index=False, header=False)

    df = pd.DataFrame(Delta_gaps)
    df.to_csv('resid_mip_gaps.csv', index=False, header=False)

    df = pd.DataFrame(solvetimes)
    df.to_csv('solvetimes.csv', index=False, header=False)

    # if cfg.theory_bounds:
    #     df = pd.DataFrame(theory_tighter_fracs)
    #     df.to_csv('theory_tighter_fracs.csv', index=False, header=False)

    # plotting resids so far
    fig, ax = plt.subplots()
    ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
    ax.plot(range(1, len(rel_LP_sols)+1), rel_LP_sols, label='LP relaxations')
    ax.plot(range(1, len(Delta_bounds)+1), Delta_bounds, label='VP bounds', linewidth=5, alpha=0.3)
    ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM', linewidth=5, alpha=0.3)

    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Fixed-point residual')
    ax.set_yscale('log')
    ax.set_title(rf'ISTA VP, $n={n}$, $m={m}$')

    ax.legend()

    plt.tight_layout()
    plt.savefig('resids.pdf')

    plt.clf()
    plt.cla()
    plt.close()

    # plotting times so far

    fig, ax = plt.subplots()
    ax.plot(range(1, len(solvetimes)+1), solvetimes, label='VP')
    # ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Solvetime (s)')
    ax.set_yscale('log')
    ax.set_title(rf'ISTA VP, $n={n}$, $m={m}$')

    ax.legend()

    plt.tight_layout()
    plt.savefig('solvetimes.pdf')

    plt.clf()
    plt.cla()
    plt.close()


def soft_threshold(x, gamma):
    return jnp.sign(x) * jax.nn.relu(jnp.abs(x) - gamma)


def ISTA_alg(At, Bt, z0, x, lambda_t, K, pnorm=1):
    n = At.shape[0]
    # yk_all = jnp.zeros((K+1, n))
    zk_all = jnp.zeros((K+1, n))
    resids = jnp.zeros(K+1)

    zk_all = zk_all.at[0].set(z0)

    def body_fun(k, val):
        zk_all, resids = val
        zk = zk_all[k]

        ykplus1 = At @ zk + Bt @ x
        zkplus1 = soft_threshold(ykplus1, lambda_t)

        if pnorm == 'inf':
            resid = jnp.max(jnp.abs(zkplus1 - zk))
        else:
            resid = jnp.linalg.norm(zkplus1 - zk, ord=pnorm)

        zk_all = zk_all.at[k+1].set(zkplus1)
        resids = resids.at[k+1].set(resid)
        return (zk_all, resids)

    zk, resids = jax.lax.fori_loop(0, K, body_fun, (zk_all, resids))
    return zk, resids


def samples(cfg, A, lambd, t, c_z, x_l, x_u):
    n = cfg.n
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    sample_idx = jnp.arange(cfg.samples.N)

    def z_sample(i):
        return c_z

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.m,), minval=x_l, maxval=x_u)

    z_samples = jax.vmap(z_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    def ista_resids(i):
        return ISTA_alg(At, Bt, z_samples[i], x_samples[i], lambda_t, cfg.K_max, pnorm=cfg.pnorm)

    _, sample_resids = jax.vmap(ista_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def samples_diffK(cfg, A, lambd, t, c_z, x_l, x_u):
    K_max = cfg.K_max

    n = cfg.n
    # t = cfg.t
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    def z_sample(i):
        return c_z

    sample_idx = jnp.arange(cfg.samples.N)
    z_samples = jax.vmap(z_sample)(sample_idx)

    maxes = []

    for k in range(1, K_max+1):
        log.info(f'computing samples for k={k}')
        def x_sample(i):
            key = jax.random.PRNGKey(cfg.samples.x_seed_offset * k + i)
            # TODO add the if, start with box case only
            return jax.random.uniform(key, shape=(cfg.m,), minval=x_l, maxval=x_u)

        x_samples_k = jax.vmap(x_sample)(sample_idx)
        def ista_resids(i):
            return ISTA_alg(At, Bt, z_samples[i], x_samples_k[i], lambda_t, k, pnorm=cfg.pnorm)

        _, sample_resids = jax.vmap(ista_resids)(sample_idx)
        log.info(sample_resids)
        max_sample_k = jnp.max(sample_resids[:, -1])
        log.info(f'max: {max_sample_k}')
        maxes.append(max_sample_k)

    return jnp.array(maxes)


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


def random_ISTA_run(cfg):
    m, n = cfg.m, cfg.n
    log.info(cfg)

    A = generate_data(cfg)
    A_eigs = jnp.real(jnp.linalg.eigvals(A.T @ A))
    log.info(f'eigenvalues of ATA: {A_eigs}')

    t = cfg.t

    log.info(f't={t}')

    if cfg.x.type == 'box':
        x_l = cfg.x.l * jnp.ones(m)
        x_u = cfg.x.u * jnp.ones(m)

    if cfg.lambd.val == 'adaptive':
        center = x_u - x_l
        lambd = cfg.lambd.scalar * jnp.max(jnp.abs(A.T @ center))
    else:
        lambd = cfg.lambd.val
    log.info(f'lambda: {lambd}')
    lambda_t = lambd * t
    log.info(f'lambda * t: {lambda_t}')

    if cfg.z0.type == 'lstsq':
        c_z = lstsq_sol(cfg, A, lambd, x_l, x_u)
    elif cfg.z0.type == 'zero':
        c_z = jnp.zeros(n)

    ISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u)


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


def sparse_coding_ISTA_run(cfg):
    # m, n = cfg.m, cfg.n
    n = cfg.n
    log.info(cfg)

    A = sparse_coding_A(cfg)

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

    ISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u)


def run(cfg):
    if cfg.problem_type == 'random':
        random_ISTA_run(cfg)
    elif cfg.problem_type == 'sparse_coding':
        sparse_coding_ISTA_run(cfg)

import logging

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

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


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3
    absA = jnp.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_upper, Ax_lower


def FISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u):

    def Init_model():
        pass

    max_sample_resids = samples(cfg, A, lambd, t, c_z, x_l, x_u)
    log.info(f'max sample resids: {max_sample_resids}')

    # pnorm = cfg.pnorm
    # m, n = cfg.m, cfg.n
    # At = jnp.eye(n) - t * A.T @ A
    # Bt = t * A.T

    # At = np.asarray(At)
    # Bt = np.asarray(Bt)
    # lambda_t = lambd * t

    # K_max = cfg.K_max

    # z_LB = jnp.zeros((K_max + 1, n))
    # z_UB = jnp.zeros((K_max + 1, n))
    # y_LB = jnp.zeros((K_max + 1, n))
    # y_UB = jnp.zeros((K_max + 1, n))
    # v_LB = jnp.zeros((K_max + 1, n))
    # v_UB = jnp.zeros((K_max + 1, n))

    # z_LB = z_LB.at[0].set(c_z)
    # z_UB = z_UB.at[0].set(c_z)
    # v_LB = w_LB.at[0].set(c_z)
    # v_UB = w_UB.at[0].set(c_z)
    # x_LB = x_l
    # x_UB = x_u


def samples(cfg, A, lambd, t, c_z, x_l, x_u):
    n = cfg.n
    # t = cfg.t
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * cfg.t

    sample_idx = jnp.arange(cfg.samples.N)

    def z_sample(i):
        return c_z

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.m,), minval=x_l, maxval=x_u)

    z_samples = jax.vmap(z_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    def fista_resids(i):
        return FISTA_alg(At, Bt, z_samples[i], x_samples[i], lambda_t, cfg.K_max, pnorm=cfg.pnorm)

    _, _, sample_resids = jax.vmap(fista_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def soft_threshold(x, gamma):
    return jnp.sign(x) * jax.nn.relu(jnp.abs(x) - gamma)


def FISTA_alg(At, Bt, z0, x, lambda_t, K, pnorm=1):
    n = At.shape[0]
    # yk_all = jnp.zeros((K+1, n))
    zk_all = jnp.zeros((K+1, n))
    wk_all = jnp.zeros((K+1, n))
    beta_all = jnp.zeros((K+1))
    resids = jnp.zeros(K+1)

    zk_all = zk_all.at[0].set(z0)
    wk_all = wk_all.at[0].set(z0)
    beta_all = beta_all.at[0].set(1.)

    def body_fun(k, val):
        zk_all, wk_all, beta_all, resids = val
        zk = zk_all[k]
        wk = wk_all[k]
        beta_k = beta_all[k]

        ykplus1 = At @ wk + Bt @ x
        zkplus1 = soft_threshold(ykplus1, lambda_t)
        beta_kplus1 = .5 * (1 + jnp.sqrt(1 + 4 * jnp.power(beta_k, 2)))
        wkplus1 = zkplus1 + (beta_k - 1) / beta_kplus1 * (zkplus1 - zk)

        if pnorm == 'inf':
            resid = jnp.max(jnp.abs(zkplus1 - zk))
        else:
            resid = jnp.linalg.norm(zkplus1 - zk, ord=pnorm)

        zk_all = zk_all.at[k+1].set(zkplus1)
        wk_all = wk_all.at[k+1].set(wkplus1)
        beta_all = beta_all.at[k+1].set(beta_kplus1)
        resids = resids.at[k+1].set(resid)

        return (zk_all, wk_all, beta_all, resids)

    zk, wk, _, resids = jax.lax.fori_loop(0, K, body_fun, (zk_all, wk_all, beta_all, resids))
    return zk, wk, resids


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


def sparse_coding_A(cfg):
    m, n = cfg.m, cfg.n
    key = jax.random.PRNGKey(cfg.A_rng_seed)

    key, subkey = jax.random.split(key)
    A = 1 / m * jax.random.normal(subkey, shape=(m, n))

    A_mask = jax.random.bernoulli(key, p=cfg.x_star.A_mask_prob, shape=(m-1, n)).astype(jnp.float64)

    masked_A = jnp.multiply(A[1:], A_mask)

    A = A.at[1:].set(masked_A)
    return A / jnp.linalg.norm(A, axis=0)


def sparse_coding_b_set(cfg, A):
    m, n = A.shape

    key = jax.random.PRNGKey(cfg.x_star.rng_seed)

    key, subkey = jax.random.split(key)
    x_star_set = jax.random.normal(subkey, shape=(n, cfg.x_star.num))

    key, subkey = jax.random.split(key)
    x_star_mask = jax.random.bernoulli(subkey, p=cfg.x_star.nonzero_prob, shape=(n, cfg.x_star.num))

    x_star = jnp.multiply(x_star_set, x_star_mask)
    # log.info(x_star)

    epsilon = cfg.x_star.epsilon_std * jax.random.normal(key, shape=(m, cfg.x_star.num))

    b_set = A @ x_star + epsilon

    # log.info(A @ x_star)
    # log.info(b_set)

    return b_set


def sparse_coding_FISTA_run(cfg):
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

    FISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u)


def random_FISTA_run(cfg):
    pass


def run(cfg):
    if cfg.problem_type == 'random':
        random_FISTA_run(cfg)
    elif cfg.problem_type == 'sparse_coding':
        sparse_coding_FISTA_run(cfg)

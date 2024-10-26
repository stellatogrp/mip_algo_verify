import logging

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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


def sparse_coding_A():
    # m, n = cfg.m, cfg.n
    m = 30
    n = 20
    A_rng_seed = 0
    A_mask_prob = 1.

    key = jax.random.PRNGKey(A_rng_seed)

    key, subkey = jax.random.split(key)
    A = 1 / m * jax.random.normal(subkey, shape=(m, n))

    A_mask = jax.random.bernoulli(key, p=A_mask_prob, shape=(m-1, n)).astype(jnp.float64)

    masked_A = jnp.multiply(A[1:], A_mask)

    A = A.at[1:].set(masked_A)
    return A / jnp.linalg.norm(A, axis=0)


def sparse_coding_b_set(A):
    m, n = A.shape

    x_star_rng_seed = 1
    x_star_num = 100
    x_star_nonzero_prob = 0.1
    key = jax.random.PRNGKey(x_star_rng_seed)

    key, subkey = jax.random.split(key)
    x_star_set = jax.random.normal(subkey, shape=(n, x_star_num))

    key, subkey = jax.random.split(key)
    x_star_mask = jax.random.bernoulli(subkey, p=x_star_nonzero_prob, shape=(n, x_star_num))

    x_star = jnp.multiply(x_star_set, x_star_mask)
    # log.info(x_star)

    epsilon_std = 0.01
    epsilon = epsilon_std * jax.random.normal(key, shape=(m, x_star_num))

    b_set = A @ x_star + epsilon

    # log.info(A @ x_star)
    # log.info(b_set)

    return b_set


def lstsq_sol(A, lambd, x_l, x_u):
    # m, n = cfg.m, cfg.n
    m, n = A.shape

    x_seed = 1
    key = jax.random.PRNGKey(x_seed)
    x_samp = jax.random.uniform(key, shape=(m,), minval=x_l, maxval=x_u)
    # log.info(x_samp)

    x_lstsq, _, _, _ = jnp.linalg.lstsq(A, x_samp)
    print(f'least squares sol: {x_lstsq}')

    z = cp.Variable(n)

    obj = cp.Minimize(.5 * cp.sum_squares(A @ z - x_samp) + lambd * cp.norm(z, 1))
    prob = cp.Problem(obj)
    prob.solve()

    print(f'lasso sol with lambda={lambd}: {z.value}')

    return x_lstsq


def sparse_coding_ISTA_run():
    # m, n = cfg.m, cfg.n
    # n = 10

    A = sparse_coding_A()

    # log.info(A)
    print(A)

    A_eigs = jnp.real(jnp.linalg.eigvals(A.T @ A))
    # log.info(f'eigenvalues of ATA: {A_eigs}')
    print(f'eigenvalues of ATA: {A_eigs}')

    L = jnp.max(A_eigs)

    # log.info(A)
    # log.info(jnp.linalg.norm(A, axis=0))

    # x_star_set = sparse_coding_x_star(cfg, A)
    b_set = sparse_coding_b_set(A)

    x_l = jnp.min(b_set, axis=1)
    x_u = jnp.max(b_set, axis=1)

    # log.info(f'size of x set: {x_u - x_l}')
    print(f'size of x set: {x_u - x_l}')

    t_rel = 1

    t = t_rel / L

    # if cfg.lambd.val == 'adaptive':
    #     center = x_u - x_l
    #     lambd = cfg.lambd.scalar * jnp.max(jnp.abs(A.T @ center))
    # else:
    #     lambd = cfg.lambd.val

    center = x_u - x_l
    lambd_scalar = 0.1
    lambd = lambd_scalar * jnp.max(jnp.abs(A.T @ center))

    print(f't={t}')
    print(f'lambda = {lambd}')
    print(f'lambda * t = {lambd * t}')

    # if cfg.z0.type == 'lstsq':
    #     c_z = lstsq_sol(cfg, A, lambd, x_l, x_u)
    # elif cfg.z0.type == 'zero':
    #     c_z = jnp.zeros(n)

    c_z = lstsq_sol(A, lambd, x_l, x_u)

    # ISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u)
    ISTA_test(A, lambd, t, c_z)


def soft_threshold(x, gamma):
    return jnp.sign(x) * jax.nn.relu(jnp.abs(x) - gamma)


def ISTA_test(A, lambd, t, c_z):
    x_test = pd.read_csv('x_heatmap.csv', header=None)
    x_test = x_test.to_numpy()
    print(x_test.shape)

    K_max = x_test.shape[1]

    n = A.shape[1]
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    all_resids = []

    for k in range(1, K_max+1):
        x_curr = x_test[:, k-1]
        print(f'--k={k}--')
        _, resids = ISTA_alg(At, Bt, c_z, x_curr, lambda_t, K_max, pnorm=jnp.inf)
        print(resids)

        all_resids.append(resids)

    all_resids = np.asarray(all_resids)
    all_resids = all_resids[:, 1:]
    # print(all_resids)
    print('---heatmap maxes---')
    print(np.max(all_resids, axis=0))


def main():
    sparse_coding_ISTA_run()


if __name__ == '__main__':
    main()

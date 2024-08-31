import logging

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


def BoundTightY(K, A, B, t, cfg, basic=True):
    n = cfg.n
    # A = jnp.zeros((n, n))
    # B = jnp.eye(n)

    # First get initial lower/upper bounds with standard techniques
    y_LB, y_UB = {}, {}
    z_LB, z_UB = {}, {}
    x_LB, x_UB = {}, {}

    if cfg.z0.type == 'zero':
        z0 = jnp.zeros(n)

    if cfg.x.type == 'box':
        xl = cfg.x.l * jnp.ones(n)
        xu = cfg.x.u * jnp.ones(n)

    for i in range(n):
        z_UB[i, 0] = z0[i]
        z_LB[i, 0] = z0[i]

    for i in range(n):
        x_LB[i] = xl[i]
        x_UB[i] = xu[i]

    for q in range(1, K+1):
        for i in range(n):
            y_UB[i, q]  = sum(A[i, j]*z_UB[j, q-1] for j in range(n) if A[i, j] > 0)
            y_UB[i, q] += sum(A[i, j]*z_LB[j, q-1] for j in range(n) if A[i, j] < 0)
            y_UB[i, q] += sum(B[i, j]*x_UB[j] for j in range(n) if B[i, j] > 0)
            y_UB[i, q] += sum(B[i, j]*x_LB[j] for j in range(n) if B[i, j] < 0)

            y_LB[i, q]  = sum(A[i, j]*z_LB[j, q-1] for j in range(n) if A[i, j] > 0)
            y_LB[i, q] += sum(A[i, j]*z_UB[j, q-1] for j in range(n) if A[i, j] < 0)
            y_LB[i, q] += sum(B[i, j]*x_LB[j] for j in range(n) if B[i, j] > 0)
            y_LB[i, q] += sum(B[i, j]*x_UB[j] for j in range(n) if B[i, j] < 0)

            # z_LB[i, q] = y_LB[i, q] if y_LB[i, q] > 0 else 0
            # z_UB[i, q] = y_UB[i, q] if y_UB[i, q] > 0 else 0
            z_LB[i, q] = jax.nn.relu(y_LB[i, q])
            z_UB[i, q] = jax.nn.relu(y_UB[i, q])

    # for q in range(1, K+1):
    #     for i in range(n):
    #         if y_LB[i, q] < 0 and y_UB[i, q] > 0:
    #             log.info((i, q))
    #             log.info(y_LB[i, q])
    #             log.info(z_LB[i, q])
    #             log.info(y_UB[i, q])
    #             log.info(z_UB[i, q])

    if basic:
        return y_LB, y_UB, z_LB, z_UB, x_LB, x_UB

    # TODO: implement advanced version


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


def NNQP_run(cfg):
    log.info(cfg)
    P = generate_P(cfg)

    if cfg.stepsize.type == 'rel':
        t = cfg.stepsize.h / cfg.L
    elif cfg.stepsize.type == 'opt':
        t = 2 / (cfg.mu + cfg. L)
    elif cfg.stepsize.type == 'abs':
        t = cfg.stepsize.h

    A = jnp.eye(cfg.n) - t * P
    B = -t * jnp.eye(cfg.n)
    K_max = cfg.K_max

    y_LB, y_UB, z_LB, z_UB, x_LB, x_UB = BoundTightY(K_max, A, B, t, cfg)


def run(cfg):
    NNQP_run(cfg)

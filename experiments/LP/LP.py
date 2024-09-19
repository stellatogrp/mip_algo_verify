import logging

import jax
import jax.numpy as jnp
import pandas as pd

jnp.set_printoptions(precision=5)  # Print few decimal places
jnp.set_printoptions(suppress=True)  # Suppress scientific notation
jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


def BoundTight():
    pass


def jax_vanilla_PDHG(A, c, t, u0, v0, x, K_max, pnorm=1):
    resids = jnp.zeros(K_max+1)

    def body_fun(i, val):
        uk, vk, resids = val
        ukplus1 = jax.nn.relu(uk - t * (c - A.T @ vk))
        vkplus1 = vk - t * (A @ (2 * ukplus1 - uk) - x)
        # if pnorm == 'inf':
        #     resids = resids.at[i].set(jnp.max(jnp.abs(znew - zk)))
        # elif pnorm == 1:
        #     resids = resids.at[i].set(jnp.linalg.norm(znew - zk, ord=pnorm))
        if pnorm == 'inf':
            resid = jnp.maximum(jnp.max(jnp.abs(ukplus1 - uk)), jnp.max(jnp.abs(vkplus1 - vk)))
        elif pnorm == 1:
            resid = jnp.linalg.norm(ukplus1 - uk, ord=pnorm) + jnp.linalg.norm(vkplus1 - vk, ord=pnorm)
        resids = resids.at[i].set(resid)
        return (ukplus1, vkplus1, resids)

    _, _, resids = jax.lax.fori_loop(1, K_max+1, body_fun, (u0, v0, resids))
    return resids


def samples(cfg, A, c, t):
    sample_idx = jnp.arange(cfg.samples.N)

    def u_sample(i):
        # if cfg.u0.type == 'zero':
        # return jnp.zeros(n)
        return jnp.zeros(cfg.n)

    def v_sample(i):
        # if cfg.v0.type == 'zero':
        # return jnp.zeros(m)
        return jnp.zeros(cfg.m)

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.m,), minval=cfg.x.l, maxval=cfg.x.u)

    u_samples = jax.vmap(u_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    def vanilla_pdhg_resids(i):
        return jax_vanilla_PDHG(A, c, t, u_samples[i], v_samples[i], x_samples[i], cfg.K_max, pnorm=cfg.pnorm)

    sample_resids = jax.vmap(vanilla_pdhg_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def LP_run(cfg, A, c, t):
    # K_max = cfg.K_max
    # K_min = cfg.K_min

    max_sample_resids = samples(cfg, A, c, t)
    log.info(max_sample_resids)

    BoundTight()


def random_LP_run(cfg):
    log.info(cfg)
    m, n = cfg.m, cfg.n
    key = jax.random.PRNGKey(cfg.rng_seed)

    key, subkey = jax.random.split(key)
    A = jax.random.normal(subkey, shape=(m, n))

    key, subkey = jax.random.split(key)
    c = jax.random.uniform(subkey, shape=(n,))

    t = cfg.stepsize
    LP_run(cfg, A, c, t)


def run(cfg):
    random_LP_run(cfg)

import logging

import gurobipy as gp
import jax
import jax.numpy as jnp

# import jax.experimental.sparse as jspa
import numpy as np
import pandas as pd

jnp.set_printoptions(precision=5)  # Print few decimal places
jnp.set_printoptions(suppress=True)  # Suppress scientific notation
jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


def BoundTight(K, A, c, t, cfg, basic=False):
    n = cfg.n
    m = cfg.m

    n_var_shape = (K+1, n)
    m_var_shape = (K+1, m)

    # First get initial lower/upper bounds with standard techniques
    utilde_LB = jnp.zeros(n_var_shape)
    utilde_UB = jnp.zeros(n_var_shape)
    u_LB = jnp.zeros(n_var_shape)
    u_UB = jnp.zeros(n_var_shape)
    v_LB = jnp.zeros(m_var_shape)
    v_UB = jnp.zeros(m_var_shape)

    if cfg.x.type == 'box':
        x_LB = cfg.x.l * jnp.ones(m)
        x_UB = cfg.x.u * jnp.ones(m)

    # TODO: atm, the jax experimental sparse pacakge does not play nice with jnp.abs for bound prop
    xC = jnp.eye(n)
    xD = t * A.T
    xE = - t * jnp.eye(n)

    # spa_xC = spa.csc_matrix(xC)
    # spa_xD = spa.csc_matrix(xD)
    # spa_xE = spa.csc_matrix(xE)
    np_A = np.asarray(A)

    vC = jnp.eye(m)
    vD = -2 * t * A
    vE = t * A
    vF = t * jnp.eye(m)

    # spa_vC = spa.csc_matrix(vC)
    # spa_vD = spa.csc_matrix(vD)
    # spa_vE = spa.csc_matrix(vE)
    # spa_vF = spa.csc_matrix(vF)

    # Bx_upper, Bx_lower = interval_bound_prop(B, x_LB, x_UB)  # only need to compute this once
    vF_x_upper, vF_x_lower = interval_bound_prop(vF, x_LB, x_UB)  # only need to compute this once
    xE_c_upper, xE_c_lower = interval_bound_prop(xE, c, c)  # if c is param, change this

    for k in range(1, K+1):
        # log.info(xC)
        # log.info(u_LB[k-1])
        xC_uk_upper, xC_uk_lower = interval_bound_prop(xC, u_LB[k-1], u_UB[k-1])
        xD_vk_upper, xD_vk_lower = interval_bound_prop(xD, v_LB[k-1], v_UB[k-1])

        utilde_LB = utilde_LB.at[k].set(xC_uk_lower + xD_vk_lower + xE_c_lower)
        utilde_UB = utilde_UB.at[k].set(xC_uk_upper + xD_vk_upper + xE_c_upper)

        u_LB = u_LB.at[k].set(jax.nn.relu(utilde_LB[k]))
        u_UB = u_UB.at[k].set(jax.nn.relu(utilde_UB[k]))

        vC_vk_upper, vC_vk_lower = interval_bound_prop(vC, v_LB[k-1], v_UB[k-1])
        vD_ukplus1_upper, vD_ukplus1_lower = interval_bound_prop(vD, u_LB[k], u_UB[k])
        vE_uk_upper, vE_uk_lower = interval_bound_prop(vE, u_LB[k-1], u_UB[k-1])
        v_LB = v_LB.at[k].set(vC_vk_lower + vD_ukplus1_lower + vE_uk_lower + vF_x_lower)
        v_UB = v_UB.at[k].set(vC_vk_upper + vD_ukplus1_upper + vE_uk_upper + vF_x_upper)

    # log.info(utilde_LB)
    # log.info(utilde_UB)
    # log.info(u_LB)
    # log.info(u_UB)
    # log.info(v_LB)
    # log.info(v_UB)
    if basic:
        return utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB

    target_var = ['u_tilde', 'v']
    for kk in range(1, K+1):
        log.info(f'^^^^^^^^ Bound tightening, K={kk} ^^^^^^^^^^')
        for target in target_var:
            if target == 'u_tilde':
                range_var = n
            else:
                range_var = m
            for ii in range(range_var):
                for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
                    model = gp.Model()
                    model.Params.OutputFlag = 0

                    utilde = model.addMVar(n_var_shape, lb=utilde_LB, ub=utilde_UB)
                    u = model.addMVar(n_var_shape, lb=u_LB, ub=u_UB)
                    v = model.addMVar(m_var_shape, lb=v_LB, ub=v_UB)
                    x = model.addMVar(m, lb=x_LB, ub=x_UB)

                    for k in range(kk):
                        # model.addConstr(y[k+1] == np.asarray(A) @ z[k] + np.asarray(B) @ x)
                        model.addConstr(utilde[k+1] == u[k] - t * (np.asarray(c) - np_A.T @ v[k]))
                        model.addConstr(v[k+1] == v[k] - t * (np_A @ (2 * u[k+1] - u[k]) - x))

                        for i in range(n):
                            if utilde_UB[k+1, i] < -0.00001:
                                model.addConstr(u[k+1, i] == 0)
                            elif utilde_LB[k+1, i] > 0.00001:
                                model.addConstr(u[k+1, i] == utilde[k+1, i])
                            else:
                                # model.addConstr(z[k+1, i] >= y[k+1, i])
                                # model.addConstr(z[k+1, i] <= y_UB[k+1, i]/ (y_UB[k+1, i] - y_LB[k+1, i]) * (y[k+1, i] - y_LB[k+1, i]))
                                model.addConstr(u[k+1, i] >= utilde[k+1, i])
                                model.addConstr(u[k+1, i] <= utilde_UB[k+1, i]/ (utilde_UB[k+1, i] - utilde_LB[k+1, i]) * (u[k+1, i] - utilde_LB[k+1, i]))

                    if target == 'u_tilde':
                        model.setObjective(utilde[kk, ii], sense)
                        model.optimize()
                    else:
                        model.setObjective(v[kk, ii], sense)
                        model.optimize()

                    if model.status != gp.GRB.OPTIMAL:
                        print('bound tighting failed, GRB model status:', model.status)
                        exit(0)
                        return None

                    obj = model.objVal
                    if target == 'u_tilde':
                        if sense == gp.GRB.MAXIMIZE:
                            utilde_UB = utilde_UB.at[kk, ii].set(min(utilde_UB[kk, ii], obj))
                            u_UB = u_UB.at[kk, ii].set(jax.nn.relu(utilde_UB[kk, ii]))
                        else:
                            utilde_LB = utilde_LB.at[kk, ii].set(min(utilde_LB[kk, ii], obj))
                            u_LB = u_LB.at[kk, ii].set(jax.nn.relu(utilde_LB[kk, ii]))
                    else: # target == 'v'
                        if sense == gp.GRB.MAXIMIZE:
                            v_UB = v_UB.at[kk, ii].set(min(v_UB[kk, ii], obj))
                        else:
                            v_LB = v_LB.at[kk, ii].set(min(v_LB[kk, ii], obj))

    log.info(jnp.all(utilde_UB - utilde_LB >= -1e-8))
    log.info(jnp.all(u_UB - u_LB >= -1e-8))
    log.info(jnp.all(v_UB - v_LB >= -1e-8))

    return utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB



def jax_vanilla_PDHG(A, c, t, u0, v0, x, K_max, pnorm=1):
    resids = jnp.zeros(K_max+1)

    def body_fun(i, val):
        uk, vk, resids = val
        ukplus1 = jax.nn.relu(uk - t * (c - A.T @ vk))
        vkplus1 = vk - t * (A @ (2 * ukplus1 - uk) - x)
        if pnorm == 'inf':
            resid = jnp.maximum(jnp.max(jnp.abs(ukplus1 - uk)), jnp.max(jnp.abs(vkplus1 - vk)))
        elif pnorm == 1:
            resid = jnp.linalg.norm(ukplus1 - uk, ord=pnorm) + jnp.linalg.norm(vkplus1 - vk, ord=pnorm)
        resids = resids.at[i].set(resid)
        return (ukplus1, vkplus1, resids)

    _, _, resids = jax.lax.fori_loop(1, K_max+1, body_fun, (u0, v0, resids))
    return resids


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3
    absA = jnp.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_upper, Ax_lower


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
    K_max = cfg.K_max
    # K_min = cfg.K_min

    max_sample_resids = samples(cfg, A, c, t)
    log.info(max_sample_resids)

    utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB = BoundTight(K_max, A, c, t, cfg, basic=cfg.basic_bounding)


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

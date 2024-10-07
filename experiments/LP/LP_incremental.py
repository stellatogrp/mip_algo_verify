import logging

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
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


def jax_vanilla_PDHG(A, c, t, u0, v0, x, K_max, pnorm=1, momentum=False, beta_func=None):
    m, n = A.shape
    uk_all = jnp.zeros((K_max+1, n))
    vk_all = jnp.zeros((K_max+1, m))
    resids = jnp.zeros(K_max+1)

    uk_all = uk_all.at[0].set(u0)
    vk_all = vk_all.at[0].set(v0)

    def body_fun(k, val):
        uk_all, vk_all, resids = val
        uk = uk_all[k]
        vk = vk_all[k]
        ukplus1 = jax.nn.relu(uk - t * (c - A.T @ vk))

        if momentum:
            ytilde_kplus1 = ukplus1 + beta_func(k) * (ukplus1 - uk)
            vkplus1 = vk - t * (A @ (2 * ytilde_kplus1 - uk) - x)
        else:
            vkplus1 = vk - t * (A @ (2 * ukplus1 - uk) - x)

        if pnorm == 'inf':
            resid = jnp.maximum(jnp.max(jnp.abs(ukplus1 - uk)), jnp.max(jnp.abs(vkplus1 - vk)))
        elif pnorm == 1:
            resid = jnp.linalg.norm(ukplus1 - uk, ord=pnorm) + jnp.linalg.norm(vkplus1 - vk, ord=pnorm)
            # resid = jnp.linalg.norm(ukplus1 - uk, ord=pnorm)
        uk_all = uk_all.at[k+1].set(ukplus1)
        vk_all = vk_all.at[k+1].set(vkplus1)
        resids = resids.at[k+1].set(resid)
        return (uk_all, vk_all, resids)

    uk, vk, resids = jax.lax.fori_loop(0, K_max, body_fun, (uk_all, vk_all, resids))
    return uk, vk, resids


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3
    absA = jnp.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_upper, Ax_lower


def samples(cfg, A, c, t, momentum=False, beta_func=None):
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
        return jax_vanilla_PDHG(A, c, t, u_samples[i], v_samples[i], x_samples[i], cfg.K_max, pnorm=cfg.pnorm,
                                momentum=momentum, beta_func=beta_func)

    _, _, sample_resids = jax.vmap(vanilla_pdhg_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def nesterov_beta_func(k):
    return k / (k + 3)


def sample_radius(cfg, A, c, t):
    sample_idx = jnp.arange(cfg.samples.init_dist_N)
    m, n = cfg.m, cfg.n

    # if cfg.u0.type == 'zero':
    #     u0 = jnp.zeros(n)

    # if cfg.v0.type == 'zero':
    #     v0 = jnp.zeros(m)

    def u_sample(i):
        return jnp.zeros(n)

    def v_sample(i):
        return jnp.zeros(m)

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.m,), minval=cfg.x.l, maxval=cfg.x.u)

    u_samples = jax.vmap(u_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    Ps = np.block([
        [1/t * np.eye(n), -A.T],
        [-A, 1/t * np.eye(m)]
    ])

    distances = jnp.zeros(cfg.samples.init_dist_N)
    for i in trange(cfg.samples.init_dist_N):
        u = cp.Variable(cfg.n)
        x_samp = x_samples[i]
        obj = cp.Minimize(c @ u)

        constraints = [A @ u == x_samp, u >= 0]
        prob = cp.Problem(obj, constraints)

        prob.solve()
        # log.info(res)
        # distances.append(jnp.linalg.norm(z.value - z_samples[i]))
        # distances = distances.at[i].set(jnp.linalg.norm(z.value - z_samples[i]))
        u_val = u.value
        v_val = -constraints[0].dual_value
        z = np.hstack([u_val, v_val])
        z0 = np.hstack([u_samples[i], v_samples[i]])
        distances = distances.at[i].set((z - z0) @ Ps @ (z - z0))

    log.info(distances)

    return jnp.max(distances), u_val, v_val, x_samp


def init_dist(cfg, A, c, t):
    m, n = cfg.m, cfg.n
    A = np.asarray(A)
    c = np.asarray(c)
    model = gp.Model()

    if cfg.x.type == 'box':
        x_LB = cfg.x.l * np.ones(m)
        x_UB = cfg.x.u * np.ones(m)

    if cfg.u0.type == 'zero':
        u0_LB = np.zeros(n)
        u0_UB = np.zeros(n)

    if cfg.v0.type == 'zero':
        v0_LB = np.zeros(m)
        v0_UB = np.zeros(m)

    bound_M = 110
    ustar = model.addMVar(n, lb=0, ub=bound_M)
    vstar = model.addMVar(m, lb=-bound_M, ub=bound_M)
    x = model.addMVar(m, lb=x_LB, ub=x_UB)
    u0 = model.addMVar(n, lb=u0_LB, ub=u0_UB)
    v0 = model.addMVar(m, lb=v0_LB, ub=v0_UB)
    w = model.addMVar(n, vtype=gp.GRB.BINARY)

    M = cfg.init_dist_M

    Ps = np.block([
        [1/t * np.eye(n), -A.T],
        [-A, 1/t * np.eye(m)]
    ])

    sample_rad, u_samp, v_samp, x_samp = sample_radius(cfg, A, c, t)
    log.info(f'sample radius: {sample_rad}')
    # exit(0)

    ustar.Start = u_samp
    vstar.Start = v_samp
    x.Start = x_samp

    # xkplus1 = np.maximum(xk - t * (c - A.T @ yk), 0)
    # ykplus1 = yk - t * (A @ (2 * xkplus1 - xk) - b)
    utilde = ustar - t * (c - A.T @ vstar)
    model.addConstr(ustar >= utilde)
    model.addConstr(vstar == vstar - t * (A @ ustar - x))
    for i in range(n):
        model.addConstr(ustar[i] <= utilde[i] + M * (1 - w[i]))
        model.addConstr(ustar[i] <= M * w[i])

    # TODO: incorporate the LP based bounding component wise
    model.addConstr(A @ ustar == x)
    model.addConstr(-A.T @ vstar + c >= 0)
    model.addConstr(c @ ustar - x @ vstar == 0)

    z0 = gp.hstack([u0, v0])
    zstar = gp.hstack([ustar, vstar])

    # obj = (ustar - u0) @ (ustar - u0)
    obj = (zstar - z0) @ Ps @ (zstar - z0)
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.optimize()

    # log.info(x.X)
    # log.info(ustar.X)
    # log.info(vstar.X)
    # log.info(f'pdhg obj val: {c @ ustar.X}')

    # log.info('testing')
    # u = cp.Variable(cfg.n)
    # x_opt = x.X
    # obj = cp.Minimize(c @ u)
    # constraints = [A @ u == x_opt, u >= 0]
    # prob = cp.Problem(obj, constraints)
    # res = prob.solve()
    # log.info(f'cvxpy u: {u.value}')
    # log.info(f'cvxpy v: {-constraints[0].dual_value}')
    # log.info(f'cvxpy obj: {res}')

    # cp_u_opt = u.value
    # cp_v_opt = -constraints[0].dual_value

    # cp_u_tilde = cp_u_opt - t * (c - A.T @ cp_v_opt)
    # cp_u = np.maximum(cp_u_tilde, 0)
    # cp_v = cp_v_opt - t * (A @ (2 * cp_u - cp_u) - x_opt)
    # log.info(cp_u)
    # log.info(cp_v)

    log.info(f'sample radius: {sample_rad}')
    log.info(f'miqp max radius: {model.objVal}')
    return model.objVal


def get_vDk_vEk(k, t, np_A, momentum=False, beta_func=None):
    vD = -2 * t * np_A
    vE = t * np_A

    if momentum:
        beta_k = beta_func(k)
        vD_k = -2 * t * (1 + beta_k) * np_A
        vE_k = t * (1 + 2 * beta_k) * np_A
    else:
        vD_k = vD
        vE_k = vE
    return vD_k, vE_k


def BoundPreprocess(K, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, cfg, A, c, t, init_C, basic_bounding=False, beta_func=None):
    m, n = cfg.m, cfg.n

    vD_k, vE_k = get_vDk_vEk(K-1, t, A, momentum=cfg.momentum)
    vC = jnp.eye(m)
    vF = t * jnp.eye(m)

    xC = jnp.eye(n)
    xD = t * A.T
    xE = - t * jnp.eye(n)

    vF_x_upper, vF_x_lower = interval_bound_prop(vF, x_LB, x_UB)  # only need to compute this once
    xE_c_upper, xE_c_lower = interval_bound_prop(xE, c, c)  # if c is param, change this

    xC_uk_upper, xC_uk_lower = interval_bound_prop(xC, u_LB[K-1], u_UB[K-1])
    xD_vk_upper, xD_vk_lower = interval_bound_prop(xD, v_LB[K-1], v_UB[K-1])

    utilde_LB = utilde_LB.at[K].set(xC_uk_lower + xD_vk_lower + xE_c_lower)
    utilde_UB = utilde_UB.at[K].set(xC_uk_upper + xD_vk_upper + xE_c_upper)

    u_LB = u_LB.at[K].set(jax.nn.relu(utilde_LB[K]))
    u_UB = u_UB.at[K].set(jax.nn.relu(utilde_UB[K]))

    vC_vk_upper, vC_vk_lower = interval_bound_prop(vC, v_LB[K-1], v_UB[K-1])
    vD_ukplus1_upper, vD_ukplus1_lower = interval_bound_prop(vD_k, u_LB[K], u_UB[K])
    vE_uk_upper, vE_uk_lower = interval_bound_prop(vE_k, u_LB[K-1], u_UB[K-1])
    v_LB = v_LB.at[K].set(vC_vk_lower + vD_ukplus1_lower + vE_uk_lower + vF_x_lower)
    v_UB = v_UB.at[K].set(vC_vk_upper + vD_ukplus1_upper + vE_uk_upper + vF_x_upper)

    # TODO: LP based and then upper bound tighten
    if basic_bounding:
        return utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB

    # log.info(u_LB)
    # log.info(u_UB)
    # log.info(v_LB)
    # log.info(v_UB)

    # now, to tighten the interval prop bound with the theory bound

    Ps = np.block([
        [1/t * np.eye(n), -A.T],
        [-A, 1/t * np.eye(m)]
    ])

    Ps_half = sp.linalg.sqrtm(Ps)
    # log.info(Ps_half)

    log.info('-computing theory bounds-')
    if K >= 2: # does not apply to the very first step
        if cfg.pnorm == 'inf':
            theory_bound = init_C / np.sqrt(K - 1)
        elif cfg.pnorm == 1:
            theory_bound = np.sqrt(n) * init_C / np.sqrt(K - 1)
        theory_model = gp.Model()
        theory_model.Params.OutputFlag = 0
        z_LB = np.hstack([u_LB[K-1], v_LB[K-1]])
        z_UB = np.hstack([u_UB[K-1], v_UB[K-1]])
        zK = theory_model.addMVar(m + n, lb=z_LB, ub=z_UB)
        zKplus1 = theory_model.addMVar(m + n, lb=-np.inf, ub=np.inf)
        theory_model.addConstr(Ps_half @ (zKplus1 - zK) <= theory_bound)
        theory_model.addConstr(Ps_half @ (zKplus1 - zK) >= -theory_bound)

        for i in range(n):
            for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
                theory_model.setObjective(zKplus1[i], sense)
                theory_model.update()
                theory_model.optimize()

                if theory_model.status != gp.GRB.OPTIMAL:
                    # print('bound tighting failed, GRB model status:', model.status)
                    log.info(f'theory bound tighting failed, GRB model status: {theory_model.status}')
                    exit(0)
                    return None

                obj = theory_model.objVal
                if sense == gp.GRB.MAXIMIZE:
                    u_UB = u_UB.at[K, i].set(min(u_UB[K, i], obj))
                else:
                    u_LB = u_LB.at[K, i].set(max(u_LB[K, i], obj))

        for i in range(m):
            for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
                theory_model.setObjective(zKplus1[n + i], sense)  # v is offset by n
                theory_model.update()
                theory_model.optimize()

                if theory_model.status != gp.GRB.OPTIMAL:
                    # print('bound tighting failed, GRB model status:', model.status)
                    log.info(f'theory bound tighting failed, GRB model status: {theory_model.status}')
                    exit(0)
                    return None

                obj = theory_model.objVal
                if sense == gp.GRB.MAXIMIZE:
                    v_UB = v_UB.at[K, i].set(min(v_UB[K, i], obj))
                else:
                    v_LB = v_LB.at[K, i].set(max(v_LB[K, i], obj))

    log.info('-computing LP based bounds-')  #TODO: finish this
    LP_model = gp.Model()
    LP_model.Params.OutputFlag = 0

    utilde = LP_model.addMVar((K+1, n), lb=utilde_LB[:K+1], ub=utilde_UB[:K+1])
    u = LP_model.addMVar((K+1, n), lb=u_LB[:K+1], ub=u_UB[:K+1])
    v = LP_model.addMVar((K+1, m), lb=v_LB[:K+1], ub=v_UB[:K+1])
    x = LP_model.addMVar(m, lb=x_LB, ub=x_UB)

    np_A = np.asarray(A)
    for k in range(1, K+1):
        LP_model.addConstr(utilde[k] == u[k-1] - t * (np.asarray(c) - np_A.T @ v[k-1]))
        if cfg.momentum:
            beta_k = beta_func(k-1)
            LP_model.addConstr(v[k] == v[k-1] - t * (np_A @ (2 * (u[k] + beta_k * (u[k] - u[k-1])) - u[k-1]) - x))
        else:
            LP_model.addConstr(v[k] == v[k-1] - t * (np_A @ (2 * u[k] - u[k-1]) - x))

    for i in range(n):
        if utilde_UB[K, i] <= 0:
            LP_model.addConstr(u[K, i] == 0)
        elif utilde_LB[K, i] > 0:
            LP_model.addConstr(u[K, i] == utilde[K, i])
        else:
            LP_model.addConstr(u[K, i] >= utilde[K, i])
            LP_model.addConstr(u[K, i] <= utilde_UB[K, i]/ (utilde_UB[K, i] - utilde_LB[K, i]) * (u[K, i] - utilde_LB[K, i]))

    target_var = ['u_tilde', 'v']
    for target in target_var:
        if target == 'u_tilde':
            range_var = n
        else:
            range_var = m
        for ii in range(range_var):
            for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
                if target == 'u_tilde':
                    LP_model.setObjective(utilde[K, ii], sense)
                    LP_model.optimize()
                else:
                    LP_model.setObjective(v[K, ii], sense)
                    LP_model.optimize()

                if LP_model.status != gp.GRB.OPTIMAL:
                    # print('bound tighting failed, GRB model status:', model.status)
                    log.info(f'bound tighting failed, GRB model status: {LP_model.status}')
                    log.info(target)
                    log.info(ii)
                    log.info(utilde_LB[K, ii])
                    log.info(utilde_UB[K, ii])

                    exit(0)
                    return None

                obj = LP_model.objVal
                if target == 'u_tilde':
                    if sense == gp.GRB.MAXIMIZE:
                        utilde_UB = utilde_UB.at[K, ii].set(min(utilde_UB[K, ii], obj))
                        u_UB = u_UB.at[K, ii].set(jax.nn.relu(utilde_UB[K, ii]))
                    else:
                        utilde_LB = utilde_LB.at[K, ii].set(max(utilde_LB[K, ii], obj))
                        u_LB = u_LB.at[K, ii].set(jax.nn.relu(utilde_LB[K, ii]))
                else: # target == 'v'
                    if sense == gp.GRB.MAXIMIZE:
                        v_UB = v_UB.at[K, ii].set(min(v_UB[K, ii], obj))
                    else:
                        v_LB = v_LB.at[K, ii].set(max(v_LB[K, ii], obj))

    return utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB


def LP_run(cfg, A, c, t):
    K_max = cfg.K_max
    # K_min = cfg.K_min
    momentum = cfg.momentum
    n, m = cfg.n, cfg.m
    pnorm = cfg.pnorm

    n_var_shape = (K_max+1, n)
    m_var_shape = (K_max+1, m)

    if cfg.beta_func == 'nesterov':
        beta_func = nesterov_beta_func

    max_sample_resids = samples(cfg, A, c, t, momentum=momentum, beta_func=beta_func)
    log.info(max_sample_resids)

    utilde_LB = -jnp.inf * jnp.ones(n_var_shape)
    utilde_UB = jnp.inf * jnp.ones(n_var_shape)
    u_LB = -jnp.inf * jnp.ones(n_var_shape)
    u_UB = jnp.inf * jnp.ones(n_var_shape)
    v_LB = -jnp.inf * jnp.ones(m_var_shape)
    v_UB = jnp.inf * jnp.ones(m_var_shape)

    if cfg.x.type == 'box':
        x_LB = cfg.x.l * jnp.ones(m)
        x_UB = cfg.x.u * jnp.ones(m)

    # TODO: if we start not at zero, change this
    utilde_LB = utilde_LB.at[0].set(0)
    utilde_UB = utilde_UB.at[0].set(0)
    u_LB = u_LB.at[0].set(0)
    u_UB = u_UB.at[0].set(0)
    v_LB = v_LB.at[0].set(0)
    v_UB = v_UB.at[0].set(0)

    init_C = init_dist(cfg, A, c, t)

    np_A = np.asarray(A)
    np_c = np.asarray(c)

    # xC = spa.eye(n)
    # xD = t * np_A.T
    # xE = -t * spa.eye(n)

    # vC = spa.eye(m)
    vD = -2 * t * np_A
    vE = t * np_A
    # vF = t * spa.eye(m)

    def get_vDk_vEk(k):
        if momentum:
            beta_k = beta_func(k)
            vD_k = -2 * t * (1 + beta_k) * np_A
            vE_k = t * (1 + 2 * beta_k) * np_A
        else:
            vD_k = vD
            vE_k = vE
        return vD_k, vE_k

    # initialize the model
    model = gp.Model()
    # model.Params.OutputFlag = 0

    # u_out = jnp.zeros(n_var_shape)
    # v_out = jnp.zeros(m_var_shape)
    # x_tracking = jnp.zeros((K_max, m))

    x = model.addMVar(m, lb=x_LB, ub=x_UB)
    u = {}
    v = {}
    w = {}

    u[0] = model.addMVar(n, lb=u_LB[0], ub=u_UB[0])
    v[0] = model.addMVar(m, lb=v_LB[0], ub=v_UB[0])

    Deltas = []
    solvetimes = []
    for K in range(1, K_max + 1):
        log.info(f'----K={K}----')
        # First do bound propagation

        utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB = BoundPreprocess(
            K, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, cfg, A, c, t, init_C, basic_bounding=cfg.basic_bounding,
        )

        u[K] = model.addMVar(n, lb=u_LB[K], ub=u_UB[K])
        v[K] = model.addMVar(m, lb=v_LB[K], ub=v_UB[K])
        w[K] = model.addMVar(n, vtype=gp.GRB.BINARY)

        vD_k, vE_k = get_vDk_vEk(K-1)
        model.addConstr(v[K] == v[K-1] + vD_k @ u[K] + vE_k @ u[K-1] + t * x)

        utilde = u[K-1] - t * (np_c - np_A.T @ v[K-1])

        for i in range(n):
            if utilde_UB[K, i] <= 0:
                model.addConstr(u[K][i] == 0)
            elif utilde_LB[K, i] > 0:
                model.addConstr(u[K][i] == utilde[i])
            else:
                model.addConstr(u[K][i] <= utilde_UB[K, i] / (utilde_UB[K, i] - utilde_LB[K, i]) * (utilde[i] - utilde_LB[K, i]))
                model.addConstr(u[K][i] >= utilde[i])
                model.addConstr(u[K][i] <= utilde[i] - utilde_LB[K, i] * (1 - w[K][i]))
                model.addConstr(u[K][i] <= utilde_UB[K, i] * w[K][i])

        model.update()

        # u_objp, u_objn, u_omega = None, None, None
        # v_objp, v_objn, v_omega = None, None, None
        # q, gamma_u, gamma_v = None, None, None
        obj_constraints = []

        if cfg.pnorm == 1 or cfg.pnorm == 'inf':
            # if K >= 2:
            #     model.remove(u_objp)
            #     model.remove(u_objn)
            #     model.remove(u_omega)

            #     model.remove(v_objp)
            #     model.remove(v_objn)
            #     model.remove(v_omega)

            #     if pnorm == 'inf':
            #         model.remove(q)
            #         model.remove(gamma_u)
            #         model.remove(gamma_v)

            for constr in obj_constraints:
                model.remove(constr)

            Uu = u_UB[K] - u_LB[K-1]
            Lu = u_LB[K] - u_UB[K-1]

            Uv = v_UB[K] - v_LB[K-1]
            Lv = v_LB[K] - v_UB[K-1]

            u_objp = model.addMVar(n, ub=jnp.abs(Uu))
            u_objn = model.addMVar(n, ub=jnp.abs(Lu))
            u_omega = model.addMVar(n, vtype=gp.GRB.BINARY)

            v_objp = model.addMVar(m, ub=jnp.abs(Uv))
            v_objn = model.addMVar(m, ub=jnp.abs(Lv))
            v_omega = model.addMVar(m, vtype=gp.GRB.BINARY)

            obj_constraints = []

            for i in range(n):
                if Lu[i] >= 0:
                    obj_constraints.append(model.addConstr(u_objp[i] == u[K][i] - u[K-1][i]))
                    obj_constraints.append(model.addConstr(u_objn[i] == 0))
                elif Uu[i] < 0:
                    obj_constraints.append(model.addConstr(u_objn[i] == u[K-1][i] - u[K][i]))
                    obj_constraints.append(model.addConstr(u_objp[i] == 0))
                else:
                    obj_constraints.append(model.addConstr(u_objp[i] - u_objn[i] == u[K][i] - u[K-1][i]))
                    obj_constraints.append(model.addConstr(u_objp[i] <= jnp.abs(Uu[i]) * u_omega[i]))
                    obj_constraints.append(model.addConstr(u_objn[i] <= jnp.abs(Lu[i]) * (1-u_omega[i])))

            for i in range(m):
                if Lv[i] >= 0:
                    obj_constraints.append(model.addConstr(v_objp[i] == v[K][i] - v[K-1][i]))
                    obj_constraints.append(model.addConstr(v_objn[i] == 0))
                elif Uv[i] < 0:
                    obj_constraints.append(model.addConstr(v_objn[i] == v[K-1][i] - v[K][i]))
                    obj_constraints.append(model.addConstr(v_objp[i] == 0))
                else:
                    obj_constraints.append(model.addConstr(v_objp[i] - v_objn[i] == v[K][i] - v[K-1][i]))
                    obj_constraints.append(model.addConstr(v_objp[i] <= jnp.abs(Uv[i]) * v_omega[i]))
                    obj_constraints.append(model.addConstr(v_objn[i] <= jnp.abs(Lv[i]) * (1-v_omega[i])))

            if pnorm == 1:
                model.setObjective(cfg.obj_scaling * (gp.quicksum(u_objp + u_objn) + gp.quicksum(v_objp + v_objn)), gp.GRB.MAXIMIZE)
            elif pnorm == 'inf':
                Mu = jnp.maximum(jnp.abs(Uu), jnp.abs(Lu))
                Mv = jnp.maximum(jnp.abs(Uv), jnp.abs(Lv))
                all_max = jnp.maximum(jnp.max(Mu), jnp.max(Mv))
                q = model.addVar(ub=all_max)
                gamma_u = model.addMVar(n, vtype=gp.GRB.BINARY)
                gamma_v = model.addMVar(m, vtype=gp.GRB.BINARY)
                for i in range(n):
                    obj_constraints.append(model.addConstr(q >= u_objp[i] + u_objn[i]))
                    obj_constraints.append(model.addConstr(q <= u_objp[i] + u_objn[i] + all_max * (1 - gamma_u[i])))

                for i in range(m):
                    obj_constraints.append(model.addConstr(q >= v_objp[i] + v_objn[i]))
                    obj_constraints.append(model.addConstr(q <= v_objp[i] + v_objn[i] + all_max * (1 - gamma_v[i])))

                obj_constraints.append(model.addConstr(gp.quicksum(gamma_u) + gp.quicksum(gamma_v) == 1))
                model.setObjective(cfg.obj_scaling * q, gp.GRB.MAXIMIZE)

            model.update()

        model.optimize()

        Deltas.append(model.objVal / cfg.obj_scaling)
        solvetimes.append(model.Runtime)
        Dk = jnp.sum(jnp.array(Deltas))

        for i in range(n):
            u_LB = u_LB.at[K, i].max(u_LB[0, i] - Dk)
            u_UB = u_UB.at[K, i].min(u_UB[0, i] + Dk)

        for i in range(m):
            v_LB = v_LB.at[K, i].max(v_LB[0, i] - Dk)
            v_UB = v_UB.at[K, i].min(v_UB[0, i] + Dk)

        log.info(Deltas)
        log.info(solvetimes)

        df = pd.DataFrame(Deltas)  # remove the first column of zeros
        if cfg.momentum:
            df.to_csv(cfg.momentum_resid_fname, index=False, header=False)
        else:
            df.to_csv(cfg.vanilla_resid_fname, index=False, header=False)

        df = pd.DataFrame(solvetimes)
        if cfg.momentum:
            df.to_csv(cfg.momentum_time_fname, index=False, header=False)
        else:
            df.to_csv(cfg.vanilla_time_fname, index=False, header=False)

        # plotting resids so far
        fig, ax = plt.subplots()
        ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
        ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

        ax.set_xlabel(r'$K$')
        ax.set_ylabel('Fixed-point residual')
        ax.set_yscale('log')
        ax.set_title(rf'PDHG VP, $n={cfg.n}$, $m={cfg.m}$')

        ax.legend()

        plt.tight_layout()

        if cfg.momentum:
            plt.savefig('momentum_resids.pdf')
        else:
            plt.savefig('vanilla_resids.pdf')

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
        ax.set_title(rf'PDHG VP, $n={cfg.n}$, $m={cfg.m}$')

        ax.legend()

        plt.tight_layout()

        if cfg.momentum:
            plt.savefig('momentum_times.pdf')
        else:
            plt.savefig('vanilla_times.pdf')
        plt.clf()
        plt.cla()
        plt.close()

    # log.info(Deltas)
    # log.info(solvetimes)


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

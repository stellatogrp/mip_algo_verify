import copy
import logging

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as spa
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


def samples(cfg, A, c, t, u0, v0, momentum=False, beta_func=None):
    sample_idx = jnp.arange(cfg.samples.N)
    m, n = A.shape

    def u_sample(i):
        return u0

    def v_sample(i):
        return v0

    x_LB, x_UB = get_x_LB_UB(cfg, A)
    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(m,), minval=x_LB, maxval=x_UB)

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


def sample_radius(cfg, A, c, t, u0, v0, x_LB, x_UB, C_norm=1):
    sample_idx = jnp.arange(cfg.samples.init_dist_N)
    m, n = A.shape

    def u_sample(i):
        return u0

    def v_sample(i):
        return v0

    x_LB, x_UB = get_x_LB_UB(cfg, A)
    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(m,), minval=x_LB, maxval=x_UB)

    u_samples = jax.vmap(u_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    Ps = np.block([
        [1/t * np.eye(n), -A.T],
        [-A, 1/t * np.eye(m)]
    ])

    Ps_half = sp.linalg.sqrtm(Ps)

    distances = jnp.zeros(cfg.samples.init_dist_N)
    for i in trange(cfg.samples.init_dist_N):
        u = cp.Variable(n)
        x_samp = x_samples[i]
        obj = cp.Minimize(c @ u)

        constraints = [A @ u == x_samp, u >= 0]
        prob = cp.Problem(obj, constraints)

        prob.solve()
        u_val = u.value
        v_val = -constraints[0].dual_value
        z = np.hstack([u_val, v_val])
        z0 = np.hstack([u_samples[i], v_samples[i]])
        if C_norm == 2:
            distances = distances.at[i].set(np.sqrt((z - z0) @ Ps @ (z - z0)))
        elif C_norm == 1:
            distances = distances.at[i].set(np.linalg.norm(Ps_half @ (z - z0), 1))

    # log.info(distances)

    return jnp.max(distances), u_val, v_val, x_samp


def init_dist(cfg, A, c, t, u0_val, v0_val, x_LB, x_UB, C_norm=1):
    SM_initC, u_samp, v_samp, x_samp = sample_radius(cfg, A, c, t, u0_val, v0_val, x_LB, x_UB, C_norm=C_norm)
    log.info(f'sample max init C: {SM_initC}')

    m, n = A.shape
    A = np.asarray(A)
    c = np.asarray(c)
    model = gp.Model()

    if cfg.x.type == 'box':
        x_LB, x_UB = get_x_LB_UB(cfg, A)

    bound_M = cfg.star_bound_M
    ustar = model.addMVar(n, lb=0, ub=bound_M)
    vstar = model.addMVar(m, lb=-bound_M, ub=bound_M)
    x = model.addMVar(m, lb=x_LB, ub=x_UB)
    u0 = model.addMVar(n, lb=u0_val, ub=u0_val)
    v0 = model.addMVar(m, lb=v0_val, ub=v0_val)
    w = model.addMVar(n, vtype=gp.GRB.BINARY)

    M = cfg.init_dist_M

    Ps = np.block([
        [1/t * np.eye(n), -A.T],
        [-A, 1/t * np.eye(m)]
    ])

    Ps_half = sp.linalg.sqrtm(Ps)

    ustar.Start = u_samp
    vstar.Start = v_samp
    x.Start = x_samp

    utilde = ustar - t * (c - A.T @ vstar)
    model.addConstr(ustar >= utilde)
    model.addConstr(vstar == vstar - t * (A @ ustar - x))
    for i in range(n):
        model.addConstr(ustar[i] <= utilde[i] + M * (1 - w[i]))
        model.addConstr(ustar[i] <= M * w[i])

    # TODO: incorporate the LP based bounding component wise
    model.addConstr(A @ ustar == x)
    model.addConstr(-A.T @ vstar + c >= 0)

    log.info(u0_val)
    log.info(v0_val)
    z0 = gp.hstack([u0, v0])
    zstar = gp.hstack([ustar, vstar])

    # obj = (ustar - u0) @ (ustar - u0)

    if C_norm == 2:
        obj = (zstar - z0) @ Ps @ (zstar - z0)
        model.setObjective(obj, gp.GRB.MAXIMIZE)
        model.optimize()

        max_rad = np.sqrt(model.objVal)

    elif C_norm == 1:
        y = Ps_half @ (z0 - zstar)
        up = model.addMVar(m + n, lb=0, ub=bound_M)
        un = model.addMVar(m + n, lb=0, ub=bound_M)
        omega = model.addMVar(m + n, vtype=gp.GRB.BINARY)

        model.addConstr(up - un == y)
        for i in range(m + n):
            model.addConstr(up[i] <= bound_M * omega[i])
            model.addConstr(un[i] <= bound_M * (1-omega[i]))

        model.setObjective(gp.quicksum(up + un), gp.GRB.MAXIMIZE)
        model.optimize()

        max_rad = model.objVal

    log.info(f'sample max init C: {SM_initC}')
    log.info(f'miqp max radius: {max_rad}')

    # log.info(x.X)
    # log.info(zstar.X)

    # log.info(jnp.linalg.norm(z0.X - zstar.X, 1))

    return max_rad


def nesterov_beta_func(k):
    return k / (k + 3)


def get_x_LB_UB(cfg, A):
    # TODO: change if we decide to use something other than a box
    if cfg.problem_type == 'flow':
        # b_tilde = np.hstack([b_supply, b_demand, u])

        flow = cfg.flow
        flow_x = flow.x

        supply_lb, supply_ub = flow_x.supply_lb, flow_x.supply_ub
        demand_lb, demand_ub = flow_x.demand_lb, flow_x.demand_ub
        capacity_lb, capacity_ub = flow_x.capacity_lb, flow_x.capacity_ub

        # log.info(A.shape)
        n_arcs = A.shape[0] - flow.n_supply - flow.n_demand
        # log.info(n_arcs)

        lb = jnp.hstack([
            supply_lb * jnp.ones(flow.n_supply),
            demand_lb * jnp.ones(flow.n_demand),
            capacity_lb * jnp.ones(n_arcs),
        ])

        ub = jnp.hstack([
            supply_ub * jnp.ones(flow.n_supply),
            demand_ub * jnp.ones(flow.n_demand),
            capacity_ub * jnp.ones(n_arcs),
        ])

        # log.info(lb)
        # log.info(ub)
    else:
        m = A.shape[0]
        lb = cfg.x.l * jnp.ones(m)
        ub = cfg.x.u * jnp.ones(m)

    return lb, ub


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


def BoundPreprocessing(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=False, beta_func=None):
    log.info(f'-interval prop for k = {k}')
    m, n = A.shape

    vD_k, vE_k = get_vDk_vEk(k-1, t, A, momentum=momentum, beta_func=beta_func)
    vC = jnp.eye(m)
    vF = t * jnp.eye(m)

    xC = jnp.eye(n)
    xD = t * A.T
    xE = - t * jnp.eye(n)

    vF_x_upper, vF_x_lower = interval_bound_prop(vF, x_LB, x_UB)  # only need to compute this once
    xE_c_upper, xE_c_lower = interval_bound_prop(xE, c, c)  # if c is param, change this

    xC_uk_upper, xC_uk_lower = interval_bound_prop(xC, u_LB[k-1], u_UB[k-1])
    xD_vk_upper, xD_vk_lower = interval_bound_prop(xD, v_LB[k-1], v_UB[k-1])

    utilde_LB = utilde_LB.at[k].set(xC_uk_lower + xD_vk_lower + xE_c_lower)
    utilde_UB = utilde_UB.at[k].set(xC_uk_upper + xD_vk_upper + xE_c_upper)

    u_LB = u_LB.at[k].set(jax.nn.relu(utilde_LB[k]))
    u_UB = u_UB.at[k].set(jax.nn.relu(utilde_UB[k]))

    vC_vk_upper, vC_vk_lower = interval_bound_prop(vC, v_LB[k-1], v_UB[k-1])
    vD_ukplus1_upper, vD_ukplus1_lower = interval_bound_prop(vD_k, u_LB[k], u_UB[k])
    vE_uk_upper, vE_uk_lower = interval_bound_prop(vE_k, u_LB[k-1], u_UB[k-1])
    v_LB = v_LB.at[k].set(vC_vk_lower + vD_ukplus1_lower + vE_uk_lower + vF_x_lower)
    v_UB = v_UB.at[k].set(vC_vk_upper + vD_ukplus1_upper + vE_uk_upper + vF_x_upper)

    return utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB


def BuildRelaxedModel(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=False, beta_func=None):
    m, n = A.shape
    model = gp.Model()
    model.Params.OutputFlag = 0
    np_c = np.asarray(c)
    np_A = np.asarray(A)

    x = model.addMVar(m, lb=x_LB, ub=x_UB)
    u = model.addMVar((k+1, n), lb=u_LB[:k+1], ub=u_UB[:k+1])
    utilde = model.addMVar((k+1, n), lb=utilde_LB[:k+1], ub=utilde_UB[:k+1])
    v = model.addMVar((k+1, m), lb=v_LB[:k+1], ub=v_UB[:k+1])

    for k in range(1, k+1):
        vD_k, vE_k = get_vDk_vEk(k-1, t, A, momentum=momentum, beta_func=beta_func)
        vD_k, vE_k = np.asarray(vD_k), np.asarray(vE_k)
        model.addConstr(v[k] == v[k-1] + vD_k @ u[k] + vE_k @ u[k-1] + t * x)
        model.addConstr(utilde[k] == u[k-1] - t * (np_c - np_A.T @ v[k-1]))

        for i in range(n):
            if utilde_UB[k, i] <= 0:
                model.addConstr(u[k, i] == 0)
            elif utilde_LB[k, i] > 0:
                model.addConstr(u[k, i] == utilde[k, i])
            else:
                model.addConstr(u[k, i] >= utilde[k, i])
                model.addConstr(u[k, i] <= utilde_UB[k, i] / (utilde_UB[k, i] - utilde_LB[k, i]) * (utilde[k, i] - utilde_LB[k, i]))
                model.addConstr(u[k, i] >= 0)

    model.update()
    return model, utilde, u, v


def theory_bound(cfg, k, A, c, t, u_LB, u_UB, v_LB, v_UB, init_C, momentum=False, beta_func=None):
    log.info(f'-theory bound for k={k}-')
    if momentum or k == 1:
        return u_LB, u_UB, v_LB, v_UB, 0

    m, n = A.shape

    Ps = np.block([
        [1/t * np.eye(n), -A.T],
        [-A, 1/t * np.eye(m)]
    ])

    Ps_half = sp.linalg.sqrtm(Ps)

    if cfg.pnorm == 'inf':
        theory_bound = init_C / np.sqrt(k - 1)
    elif cfg.pnorm == 1:
        theory_bound = np.sqrt(n) * init_C / np.sqrt(k - 1)

    model = gp.Model()
    model.setParam('MIPGap', cfg.mipgap)
    model.setParam('TimeLimit', cfg.timelimit)

    # model.Params.OutputFlag = 0
    z_LB = np.hstack([u_LB[k-1], v_LB[k-1]])
    z_UB = np.hstack([u_UB[k-1], v_UB[k-1]])
    zK = model.addMVar(m + n, lb=z_LB, ub=z_UB)
    zKplus1 = model.addMVar(m + n, lb=-np.inf, ub=np.inf)
    model.addConstr(Ps_half @ (zKplus1 - zK) <= theory_bound)
    model.addConstr(Ps_half @ (zKplus1 - zK) >= -theory_bound)

    theory_tight_count = 0
    for i in range(n):
        for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
            model.setObjective(zKplus1[i], sense)
            model.update()
            model.optimize()

            if model.status != gp.GRB.OPTIMAL:
                # print('bound tighting failed, GRB model status:', model.status)
                log.info(f'theory bound tighting failed, GRB model status: {model.status}')
                exit(0)
                return None

            obj = jax.nn.relu(model.objVal)
            if sense == gp.GRB.MAXIMIZE:
                if obj < u_UB[k, i]:
                    theory_tight_count += 1
                u_UB = u_UB.at[k, i].set(min(u_UB[k, i], obj))
            else:
                if obj > u_LB[k, i]:
                    theory_tight_count += 1
                u_LB = u_LB.at[k, i].set(max(u_LB[k, i], obj))

    for i in range(m):
        for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
            model.setObjective(zKplus1[n + i], sense)  # v is offset by n
            model.update()
            model.optimize()

            if model.status != gp.GRB.OPTIMAL:
                # print('bound tighting failed, GRB model status:', model.status)
                log.info(f'theory bound tighting failed, GRB model status: {model.status}')
                exit(0)
                return None

            obj = model.objVal
            if sense == gp.GRB.MAXIMIZE:
                if obj < v_UB[k, i]:
                    theory_tight_count += 1
                v_UB = v_UB.at[k, i].set(min(v_UB[k, i], obj))
            else:
                if obj > v_LB[k, i]:
                    theory_tight_count += 1
                v_LB = v_LB.at[k, i].set(max(v_LB[k, i], obj))

    return u_LB, u_UB, v_LB, v_UB, theory_tight_count / (m + n)


def compute_lI(w, x, b, Lhat, Uhat, I, Icomp):
    if I.shape[0] == 0:
        return jnp.sum(jnp.multiply(w, Uhat)) + b
    if Icomp.shape[0] == 0:
        return jnp.sum(jnp.multiply(w, Lhat)) + b

    w_I = w[I]
    w_Icomp = w[Icomp]

    Lhat_I = Lhat[I]
    Uhat_I = Uhat[Icomp]

    return jnp.sum(jnp.multiply(w_I, Lhat_I)) + jnp.sum(jnp.multiply(w_Icomp, Uhat_I)) + b


def compute_v(wi, xi, b, Lhat, Uhat):
    idx = jnp.arange(wi.shape[0])
    # log.info(idx)

    filtered_idx = jnp.array([j for j in idx if wi[j] != 0 and jnp.abs(Uhat[j] - Lhat[j]) > 1e-7])
    # log.info(filtered_idx)

    def key_func(j):
        return (xi[j] - Lhat[j]) / (Uhat[j] - Lhat[j])

    keys = jnp.array([key_func(j) for j in filtered_idx])
    # log.info(keys)
    sorted_idx = jnp.argsort(keys)
    filtered_idx = filtered_idx[sorted_idx]

    # log.info(filtered_idx)

    I = jnp.array([])
    Icomp = set(range(wi.shape[0]))

    # log.info(Icomp)

    lI = compute_lI(wi, xi, b, Lhat, Uhat, I, jnp.array(list(Icomp)))
    log.info(f'original lI: {lI}')
    if lI < 0:
        return None, None, None, None

    # for h in filtered_idx:
    #     Itest = jnp.append(I, h)
    #     Icomp.remove(int(h))

    #     lI = compute_lI(wi, xi, b, Lhat, Uhat, Itest.astype(jnp.integer), jnp.array(list(Icomp)))  # TODO: check lI calc, needs to be calced before adding h
    #     # log.info(lI)  # the returned lI needs to be nonnegative
    #     if lI < 0:
    #         Iint = I.astype(jnp.integer)
    #         log.info(f'h={h}')
    #         rhs = jnp.sum(jnp.multiply(wi[Iint], xi[Iint])) + lI / (Uhat[int(h)] - Lhat[int(h)]) * (xi[int(h)] - Lhat[int(h)])
    #         return Iint, rhs, lI, int(h)
    #         break  # TODO: add what happens if loop breaks by reaching end of array

    #     I = Itest
    for h in filtered_idx:
        Itest = jnp.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        log.info(Itest)
        log.info(Icomp_test)

        lI_new = compute_lI(wi, xi, b, Lhat, Uhat, Itest.astype(jnp.integer), jnp.array(list(Icomp_test)))
        log.info(lI_new)
        if lI_new < 0:
            Iint = I.astype(jnp.integer)
            log.info(f'h={h}')
            log.info(f'lI before and after: {lI}, {lI_new}')
            rhs = jnp.sum(jnp.multiply(wi[Iint], xi[Iint])) + lI / (Uhat[int(h)] - Lhat[int(h)]) * (xi[int(h)] - Lhat[int(h)])
            return Iint, rhs, lI, int(h)

        I = Itest
        Icomp = Icomp_test
        lI = lI_new
    else:
        return None, None, None, None


def add_conv_cuts(cfg, k, i, sense, A, c, t, u_LB, u_UB, v_LB, v_UB, u, v, u_out):
    log.info(f'(k,i) = {(k, i)}')
    log.info(f'sense={sense}')
    m, n = A.shape
    L_hat = jnp.zeros((m + n))
    U_hat = jnp.zeros((m + n))

    ukminus1_LB = u_LB[k-1]
    ukminus1_UB = u_UB[k-1]
    vkminus1_LB = v_LB[k-1]
    vkminus1_UB = v_UB[k-1]

    Ci = jnp.eye(n)[i]
    Di = t * A.T[i]
    minustci = -t * c[i]

    # xi = jnp.hstack([u.X[k-1], v.X[k-1]])

    xi = jnp.hstack([u, v])
    wi = jnp.hstack([Ci, Di])

    for j in range(n):
        if Ci[j] >= 0:
            L_hat = L_hat.at[j].set(ukminus1_LB[j])
            U_hat = U_hat.at[j].set(ukminus1_UB[j])
        else:
            L_hat = L_hat.at[j].set(ukminus1_UB[j])
            U_hat = U_hat.at[j].set(ukminus1_LB[j])

    for j in range(m):
        if Di[j] >= 0:
            L_hat = L_hat.at[n + j].set(vkminus1_LB[j])
            U_hat = U_hat.at[n + j].set(vkminus1_UB[j])
        else:
            L_hat = L_hat.at[n + j].set(vkminus1_UB[j])
            U_hat = U_hat.at[n + j].set(vkminus1_LB[j])

    Iint, rhs, lI, h = compute_v(wi, xi, minustci, L_hat, U_hat)

    if Iint is None:
        return None, None, None, None, None

    log.info(f'rhs:{rhs}')
    # log.info(f'lhs:{u.X[k, i]}')
    log.info(f'lhs:{u_out[i]}')
    log.info(f'lI: {lI}')
    # lhs = u.X[k, i]
    lhs = u_out[i]
    if lhs > rhs + 1e-6:
        log.info('found a violated cut')
        log.info(f'with lI = {lI}')
        log.info(f'and I = {Iint}')
        # exit(0)

    if lhs > rhs + 1e-6:
        return Iint, lI, h, L_hat, U_hat
    else:
        return None, None, None, None, None


def create_new_constr(A, k, i, t, Iint, lI, h, u, v, Lhat, Uhat):
    n = A.shape[1]
    Ci = jnp.eye(n)[i]
    Di = t * A.T[i]

    w = jnp.hstack([Ci, Di])
    x = gp.hstack([u[k-1], v[k-1]])

    new_constr = 0
    for idx in Iint:
        new_constr += w[idx] * (x[idx] - Lhat[idx])
    new_constr += lI / (Uhat[h] - Lhat[h]) * (x[h] - Lhat[h])

    # return u[k, i] <= new_constr
    return u[k][i] <= new_constr


def BoundTightU(cfg, k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=False, beta_func=None):
    log.info(f'-LP based bounds for k = {k} on u-')
    model, utilde, u, v = BuildRelaxedModel(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)
    m, n = A.shape

    for sense in [gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE]:
        for i in range(n):

            model.setObjective(utilde[k, i], sense)
            model.update()
            model.optimize()

            if model.status != gp.GRB.OPTIMAL:
                print('bound tighting failed, GRB model status:', model.status)
                exit(0)
                return None

            # if cfg.exact_conv_relax and k == 6 and sense == -1:
            old_objval = model.objVal
            if cfg.exact_conv_relax.use_in_bounds:
                if utilde_UB[k, i] > 0 and utilde_LB[k, i] < 0:
                    Iint, lI, h, Lhat, Uhat = add_conv_cuts(cfg, k, i, sense, A, c, t, u_LB, u_UB, v_LB, v_UB, u[k-1].X, v[k-1].X, u[k].X)
                    if Iint is not None:
                        log.info(Iint)
                        log.info(h)
                        log.info(lI)
                        new_constr = create_new_constr(A, k, i, t, Iint, lI, h, u, v, Lhat, Uhat)
                        model.addConstr(new_constr)
                        model.update()
                        model.optimize()
                        new_objval = model.objVal
                        log.info(f'sense={sense}')
                        log.info(f'old_objval: {old_objval}')
                        log.info(f'new_objval: {new_objval}')
                        if jnp.abs(new_objval - old_objval) >= 1e-6:
                            log.info('large change')
                            exit(0)

            if sense == gp.GRB.MAXIMIZE:
                utilde_UB = utilde_UB.at[k, i].set(old_objval)
            else:
                utilde_LB = utilde_LB.at[k, i].set(old_objval)

            if utilde_LB[k, i] > utilde_UB[k, i]:
                raise ValueError('Infeasible bounds', sense, i, k, utilde_LB[k, i], utilde_UB[k, i])

    u_UB = u_UB.at[k].set(jax.nn.relu(utilde_UB[k]))
    u_LB = u_LB.at[k].set(jax.nn.relu(utilde_LB[k]))
    return utilde_LB, utilde_UB, u_LB, u_UB


def BoundTightV(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=False, beta_func=None):
    log.info(f'-LP based bounds for k = {k} on v-')
    model, _, _, v = BuildRelaxedModel(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)
    m = A.shape[0]

    for sense in [gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE]:
        for i in range(m):
            model.setObjective(v[k, i], sense)
            model.update()
            model.optimize()

            if model.status != gp.GRB.OPTIMAL:
                print('bound tighting failed, GRB model status:', model.status)
                exit(0)
                return None

            if sense == gp.GRB.MAXIMIZE:
                v_UB = v_UB.at[k, i].set(model.objVal)
            else:
                v_LB = v_LB.at[k, i].set(model.objVal)

            if v_LB[k, i] > v_UB[k, i] + 1e-6:
                raise ValueError('Infeasible bounds', sense, i, k, v_LB[k, i], v_UB[k, i])

    return v_LB, v_UB


def LP_run(cfg, A, c, t, u0, v0):

    def Init_model():
        model = gp.Model()
        model.setParam('MIPGap', cfg.mipgap)
        model.setParam('TimeLimit', cfg.timelimit)
        model.setParam('MIPFocus', cfg.mipfocus)

        x = model.addMVar(m, lb=x_LB, ub=x_UB)
        u[0] = model.addMVar(n, lb=u0, ub=u0)  # if nonsingleton, change here
        v[0] = model.addMVar(m, lb=v0, ub=v0)

        model.update()
        return model, x

    def ModelNextStep(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=False, beta_func=None, obj_scaling=cfg.obj_scaling.default):
        obj_constraints = []
        np_c = np.asarray(c)
        np_A = np.asarray(A)

        # utilde[k] = model.addMVar(n, lb=utilde_LB[k], ub=utilde_UB[k])
        u[k] = model.addMVar(n, lb=u_LB[k], ub=u_UB[k])
        v[k] = model.addMVar(m, lb=v_LB[k], ub=v_UB[k])

        # for constr in obj_constraints:
        #     model.remove(constr)
        # model.update()

        vD_k, vE_k = get_vDk_vEk(k-1, t, A, momentum=momentum, beta_func=beta_func)
        vD_k, vE_k = np.asarray(vD_k), np.asarray(vE_k)

        # affine constraints
        model.addConstr(v[k] == v[k-1] + vD_k @ u[k] + vE_k @ u[k-1] + t * x)
        utilde = u[k-1] - t * (np_c - np_A.T @ v[k-1])

        for i in range(n):
            if utilde_UB[k, i] <= 0:
                model.addConstr(u[k][i] == 0)
            elif utilde_LB[k, i] > 0:
                model.addConstr(u[k][i] == utilde[i])
            else:
                w[k, i] = model.addVar(vtype=gp.GRB.BINARY)
                model.addConstr(u[k][i] >= utilde[i])
                model.addConstr(u[k][i] <= utilde_UB[k, i] / (utilde_UB[k, i] - utilde_LB[k, i]) * (utilde[i] - utilde_LB[k, i]))
                model.addConstr(u[k][i] <= utilde[i] - utilde_LB[k, i] * (1 - w[k, i]))
                model.addConstr(u[k][i] <= utilde_UB[k, i] * w[k, i])

        # setting up for objective
        Uu = u_UB[k] - u_LB[k-1]
        Lu = u_LB[k] - u_UB[k-1]

        Uv = v_UB[k] - v_LB[k-1]
        Lv = v_LB[k] - v_UB[k-1]

        u_objp = model.addMVar(n, ub=jnp.abs(Uu))
        u_objn = model.addMVar(n, ub=jnp.abs(Lu))
        u_omega = model.addMVar(n, vtype=gp.GRB.BINARY)

        v_objp = model.addMVar(m, ub=jnp.abs(Uv))
        v_objn = model.addMVar(m, ub=jnp.abs(Lv))
        v_omega = model.addMVar(m, vtype=gp.GRB.BINARY)

        if pnorm == 1 or pnorm == 'inf':
            for i in range(n):
                if Lu[i] >= 0:
                    obj_constraints.append(model.addConstr(u_objp[i] == u[k][i] - u[k-1][i]))
                    obj_constraints.append(model.addConstr(u_objn[i] == 0))
                elif Uu[i] < 0:
                    obj_constraints.append(model.addConstr(u_objn[i] == u[k-1][i] - u[k][i]))
                    obj_constraints.append(model.addConstr(u_objp[i] == 0))
                else:
                    obj_constraints.append(model.addConstr(u_objp[i] - u_objn[i] == u[k][i] - u[k-1][i]))
                    obj_constraints.append(model.addConstr(u_objp[i] <= jnp.abs(Uu[i]) * u_omega[i]))
                    obj_constraints.append(model.addConstr(u_objn[i] <= jnp.abs(Lu[i]) * (1-u_omega[i])))

            for i in range(m):
                if Lv[i] >= 0:
                    obj_constraints.append(model.addConstr(v_objp[i] == v[k][i] - v[k-1][i]))
                    obj_constraints.append(model.addConstr(v_objn[i] == 0))
                elif Uv[i] < 0:
                    obj_constraints.append(model.addConstr(v_objn[i] == v[k-1][i] - v[k][i]))
                    obj_constraints.append(model.addConstr(v_objp[i] == 0))
                else:
                    obj_constraints.append(model.addConstr(v_objp[i] - v_objn[i] == v[k][i] - v[k-1][i]))
                    obj_constraints.append(model.addConstr(v_objp[i] <= jnp.abs(Uv[i]) * v_omega[i]))
                    obj_constraints.append(model.addConstr(v_objn[i] <= jnp.abs(Lv[i]) * (1-v_omega[i])))

            if pnorm == 1:
                model.setObjective(1 / obj_scaling * (gp.quicksum(u_objp + u_objn) + gp.quicksum(v_objp + v_objn)), gp.GRB.MAXIMIZE)
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
                model.setObjective(1 / obj_scaling * q, gp.GRB.MAXIMIZE)

        model.update()
        # log.info(u_omega)
        if cfg.exact_conv_relax.use_in_l1_rel:
            rel_model = model.relax()
            rel_model.optimize()
            log.info(f'relaxed obj val at {k}: {rel_model.objVal}')
            rel_u = np.array([])
            rel_v = np.array([])

            rel_u_out = np.array([])

            for var in u[k-1]:
                # log.info(var.VarName)
                # log.info(var.VarName.item())
                rel_u = np.append(rel_u, rel_model.getVarByName(var.VarName.item()).X)

            for var in u[k]:
                rel_u_out = np.append(rel_u_out, rel_model.getVarByName(var.VarName.item()).X)

            for var in v[k-1]:
                rel_v = np.append(rel_v, rel_model.getVarByName(var.VarName.item()).X)

            log.info(rel_u)
            log.info(rel_v)

            for i in range(n):
                # log.info(f'(k, i): {(k, i)}')
                sense = 1
                Iint, lI, h, L_hat, U_hat = add_conv_cuts(cfg, k, i, sense, A, c, t, u_LB, u_UB, v_LB, v_UB, rel_u, rel_v, rel_u_out)

                if Iint is not None:
                    log.info('new constraint added')
                    model.addConstr(create_new_constr(A, k, i, t, Iint, lI, h, u, v, L_hat, U_hat))
        model.update()
        model.optimize()

        for constr in obj_constraints:
            try:
                model.remove(constr)
            except gp.GurobiError:
                pass

        # model.objBound * obj_scaling, model.MIPGap
        return model.objVal * obj_scaling, model.objBound * obj_scaling, model.MIPGap, model.Runtime, x.X

    log.info(cfg)

    K_max = cfg.K_max
    # K_min = cfg.K_min
    momentum = cfg.momentum
    m, n = A.shape
    pnorm = cfg.pnorm

    n_var_shape = (K_max+1, n)
    m_var_shape = (K_max+1, m)

    if cfg.beta_func == 'nesterov':
        beta_func = nesterov_beta_func

    max_sample_resids = samples(cfg, A, c, t, u0, v0, momentum=momentum, beta_func=beta_func)
    log.info(max_sample_resids)

    if cfg.x.type == 'box':
        x_LB, x_UB = get_x_LB_UB(cfg, A)

    utilde_LB = jnp.zeros(n_var_shape)
    utilde_UB = jnp.zeros(n_var_shape)
    u_LB = jnp.zeros(n_var_shape)
    u_UB = jnp.zeros(n_var_shape)
    v_LB = jnp.zeros(m_var_shape)
    v_UB = jnp.zeros(m_var_shape)

    u_LB = u_LB.at[0].set(u0)
    u_UB = u_UB.at[0].set(u0)
    v_LB = v_LB.at[0].set(v0)
    v_UB = v_UB.at[0].set(v0)

    init_C = init_dist(cfg, A, c, t, u0, v0, x_LB, x_UB, C_norm=cfg.C_norm)
    # init_C = 1e4

    # utilde, u, v = {}, {}, {}
    u, v = {}, {}
    w = {}

    model, x = Init_model()

    Deltas = []
    Delta_bounds = []
    Delta_gaps = []
    solvetimes = []
    theory_tighter_fracs = []

    obj_scaling = cfg.obj_scaling.default

    x_out = jnp.zeros((K_max, m))
    for k in range(1, K_max+1):
        log.info(f'----K={k}----')
        utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB = BoundPreprocessing(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)

        if jnp.any(utilde_LB > utilde_UB):
            raise AssertionError('utilde bounds invalid after interval prop')
        if jnp.any(u_LB > u_UB):
            raise AssertionError('u bounds invalid after interval prop')
        if jnp.any(v_LB > v_UB):
            raise AssertionError('v bounds invalid after interval prop')

        if cfg.theory_bounds:
            u_LB, u_UB, v_LB, v_UB, theory_tight_frac = theory_bound(cfg, k, A, c, t, u_LB, u_UB, v_LB, v_UB, init_C, momentum=momentum, beta_func=beta_func)
            theory_tighter_fracs.append(theory_tight_frac)

        if cfg.opt_based_tightening:
            for _ in range(cfg.num_obbt_iter):
                utilde_LB, utilde_UB, u_LB, u_UB = BoundTightU(cfg, k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)
                v_LB, v_UB = BoundTightV(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)

        if jnp.any(utilde_LB > utilde_UB):
            raise AssertionError('utilde bounds invalid after LP based bounds')
        if jnp.any(u_LB > u_UB):
            raise AssertionError('u bounds invalid after LP based bounds')
        if jnp.any(v_LB > v_UB):
            raise AssertionError('v bounds invalid after LP based bounds')

        result, bound, opt_gap, time, xval = ModelNextStep(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func, obj_scaling=obj_scaling)
        x_out = x_out.at[k-1].set(xval)
        log.info(result)
        log.info(xval)

        Deltas.append(result)
        Delta_bounds.append(bound)
        Delta_gaps.append(opt_gap)
        solvetimes.append(time)

        if cfg.obj_scaling.val == 'adaptive':
            obj_scaling = result

        log.info(Deltas)
        log.info(solvetimes)
        log.info(theory_tighter_fracs)

        if cfg.postprocessing:
            # Dk = jnp.sum(jnp.array(Deltas))
            Dk = jnp.sum(jnp.array(Delta_bounds))
            for i in range(n):
                u_LB = u_LB.at[k, i].set(max(u0[i] - Dk, jax.nn.relu(utilde_LB[k, i])))
                u_UB = u_UB.at[k, i].set(min(u0[i] + Dk, jax.nn.relu(utilde_UB[k, i])))
                u[k][i].LB = u_LB[k, i]
                u[k][i].UB = u_UB[k, i]

        for i in range(m):
            v_LB = v_LB.at[k, i].set(max(v0[i] - Dk, v_LB[k, i]))
            v_UB = v_UB.at[k, i].set(min(v0[i] + Dk, v_UB[k, i]))
            v[k][i].LB = v_LB[k, i]
            v[k][i].UB = v_UB[k, i]

        model.update()

        df = pd.DataFrame(Deltas)  # remove the first column of zeros
        if cfg.momentum:
            df.to_csv(cfg.momentum_resid_fname, index=False, header=False)
        else:
            df.to_csv(cfg.vanilla_resid_fname, index=False, header=False)

        df = pd.DataFrame(Delta_bounds)
        df.to_csv('resid_bounds.csv', index=False, header=False)

        df = pd.DataFrame(Delta_gaps)
        df.to_csv('resid_mip_gaps.csv', index=False, header=False)

        df = pd.DataFrame(solvetimes)
        if cfg.momentum:
            df.to_csv(cfg.momentum_time_fname, index=False, header=False)
        else:
            df.to_csv(cfg.vanilla_time_fname, index=False, header=False)

        if cfg.theory_bounds:
            df = pd.DataFrame(theory_tighter_fracs)
            df.to_csv('theory_tighter_fracs.csv', index=False, header=False)

        # plotting resids so far
        fig, ax = plt.subplots()
        ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
        ax.plot(range(1, len(Delta_bounds)+1), Delta_bounds, label='VP bounds', linewidth=5, alpha=0.3)
        ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM', linewidth=5, alpha=0.3)

        ax.set_xlabel(r'$K$')
        ax.set_ylabel('Fixed-point residual')
        ax.set_yscale('log')
        ax.set_title(rf'PDHG VP, $n={n}$, $m={m}$')

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
        ax.set_title(rf'PDHG VP, $n={n}$, $m={m}$')

        ax.legend()

        plt.tight_layout()

        if cfg.momentum:
            plt.savefig('momentum_times.pdf')
        else:
            plt.savefig('vanilla_times.pdf')
        plt.clf()
        plt.cla()
        plt.close()

        # log.info('xvals:')
        # log.info(x_out)

        x_out_plot = x_out.T

        if cfg.problem_type == 'flow':
            x_out_plot = x_out_plot[cfg.flow.n_supply: cfg.flow.n_supply + cfg.flow.n_demand]

        plt.imshow(x_out_plot, cmap='viridis')
        plt.colorbar()

        plt.xlabel(r'$K$')
        plt.savefig('x_heatmap.pdf')

        df = pd.DataFrame(x_out_plot)
        df.to_csv('x_heatmap.csv', index=False, header=False)

        plt.clf()
        plt.cla()
        plt.close()

def random_LP_run(cfg):
    log.info(cfg)
    m, n = cfg.m, cfg.n
    key = jax.random.PRNGKey(cfg.rng_seed)

    key, subkey = jax.random.split(key)
    A = jax.random.normal(subkey, shape=(m, n))

    key, subkey = jax.random.split(key)
    c = jax.random.uniform(subkey, shape=(n,))

    # t = cfg.stepsize
    t = cfg.rel_stepsize / jnp.linalg.norm(A, ord=2)

    if cfg.u0.type == 'zero':
        u0 = jnp.zeros(n)

    if cfg.v0.type == 'zero':
        v0 = jnp.zeros(m)

    LP_run(cfg, A, c, t, u0, v0)


def mincostflow_LP_run(cfg):
    log.info(cfg)
    flow = cfg.flow
    n_supply, n_demand, p, seed = flow.n_supply, flow.n_demand, flow.p, flow.seed

    G = nx.bipartite.random_graph(n_supply, n_demand, p, seed=seed, directed=False)
    A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)

    n_arcs = A.shape[1]
    A[n_supply:, :] *= -1

    log.info(A.todense())

    t = cfg.rel_stepsize / spa.linalg.norm(A, ord=2)
    log.info(f'using t={t}')

    key = jax.random.PRNGKey(flow.c.seed)
    c = jax.random.uniform(key, shape=(n_arcs,), minval=flow.c.low, maxval=flow.c.high)
    log.info(c)

    A_supply = A[:n_supply, :]
    A_demand = A[n_supply:, :]

    A_block = spa.bmat([
        [A_supply, spa.eye(n_supply), None],
        [A_demand, None, None],
        [spa.eye(n_arcs), None, spa.eye(n_arcs)]
    ])

    log.info(f'overall A size: {A_block.shape}')

    n_tilde = A_block.shape[1]
    c_tilde = np.zeros(n_tilde)
    c_tilde[:n_arcs] = c

    log.info(c_tilde)

    m, n = A_block.shape
    if flow.u0.type == 'high_demand':
        # x, _ = get_x_LB_UB(cfg, A_block)  # use the lower bound
        supply_lb = flow.x.supply_lb
        demand_lb = flow.x.demand_lb  # demand in our convention is negative so use the lower bound
        capacity_ub = flow.x.capacity_ub

        b_tilde = jnp.hstack([
            supply_lb * jnp.ones(flow.n_supply),
            demand_lb * jnp.ones(flow.n_demand),
            capacity_ub * jnp.ones(n_arcs),
        ])
        log.info(f'hardest x to satisfy: {b_tilde}')

        x_tilde = cp.Variable(n)

        constraints = [A_block @ x_tilde == b_tilde, x_tilde >= 0]

        prob = cp.Problem(cp.Minimize(c_tilde.T @ x_tilde), constraints)
        res = prob.solve()
        log.info(res)

        if res == np.inf:
            log.info('the problem in the family with lowest supply and highest demand is infeasible')
            exit(0)

        u0 = x_tilde.value
        v0 = constraints[0].dual_value
        # v0 = jnp.zeros(m)

        log.info(f'u0: {u0}')
        log.info(f'v0: {v0}')

    else:
        if flow.u0.type == 'zero':
            u0 = jnp.zeros(n)
            # u0 = jnp.ones(n)  # TODO change back to zeros when done testing

        if flow.v0.type == 'zero':
            v0 = jnp.zeros(m)

    LP_run(cfg, jnp.asarray(A_block.todense()), c_tilde, t, u0, v0)


def run(cfg):
    if cfg.problem_type == 'flow':
        mincostflow_LP_run(cfg)
    else:
        random_LP_run(cfg)

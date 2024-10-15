import logging

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
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


def sample_radius(cfg, A, c, t, u0, v0):
    sample_idx = jnp.arange(cfg.samples.init_dist_N)
    m, n = A.shape

    # if cfg.u0.type == 'zero':
    #     u0 = jnp.zeros(n)

    # if cfg.v0.type == 'zero':
    #     v0 = jnp.zeros(m)

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
        distances = distances.at[i].set(np.sqrt((z - z0) @ Ps @ (z - z0)))

    log.info(distances)

    return jnp.max(distances), u_val, v_val, x_samp


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
    return model, utilde, v


def BoundTightU(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=False, beta_func=None):
    log.info(f'-LP based bounds for k = {k} on u-')
    model, utilde, _ = BuildRelaxedModel(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)
    n = A.shape[1]
    for sense in [gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE]:
        for i in range(n):
            model.setObjective(utilde[k, i], sense)
            model.update()
            model.optimize()

            if model.status != gp.GRB.OPTIMAL:
                print('bound tighting failed, GRB model status:', model.status)
                exit(0)
                return None

            if sense == gp.GRB.MAXIMIZE:
                utilde_UB = utilde_UB.at[k, i].set(model.objVal)
            else:
                utilde_LB = utilde_LB.at[k, i].set(model.objVal)

            if utilde_LB[k, i] > utilde_UB[k, i]:
                raise ValueError('Infeasible bounds', sense, i, k, utilde_LB[k, i], utilde_UB[k, i])

    u_UB = u_UB.at[k].set(jax.nn.relu(utilde_UB[k]))
    u_LB = u_LB.at[k].set(jax.nn.relu(utilde_LB[k]))
    return utilde_LB, utilde_UB, u_LB, u_UB


def BoundTightV(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=False, beta_func=None):
    log.info(f'-LP based bounds for k = {k} on v-')
    model, _, v = BuildRelaxedModel(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)
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

            if v_LB[k, i] > v_UB[k, i]:
                raise ValueError('Infeasible bounds', sense, i, k, v_LB[k, i], v_UB[k, i])

    return v_LB, v_UB


def LP_run(cfg, A, c, t, u0, v0):

    def Init_model():
        model = gp.Model()

        x = model.addMVar(m, lb=x_LB, ub=x_UB)
        u[0] = model.addMVar(n, lb=u0, ub=u0)  # if nonsingleton, change here
        v[0] = model.addMVar(m, lb=v0, ub=v0)

        model.update()
        return model, x

    def ModelNextStep(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=False, beta_func=None):
        obj_constraints = []
        np_c = np.asarray(c)
        np_A = np.asarray(A)

        # utilde[k] = model.addMVar(n, lb=utilde_LB[k], ub=utilde_UB[k])
        u[k] = model.addMVar(n, lb=u_LB[k], ub=u_UB[k])
        v[k] = model.addMVar(m, lb=v_LB[k], ub=v_UB[k])

        for constr in obj_constraints:
            model.remove(constr)
        model.update()

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

        return model.objVal / cfg.obj_scaling, model.Runtime

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

    # utilde, u, v = {}, {}, {}
    u, v = {}, {}
    w = {}

    model, x = Init_model()

    Deltas = []
    solvetimes = []
    for k in range(1, K_max+1):
        log.info(f'----K={k}----')
        utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB = BoundPreprocessing(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)

        if jnp.any(utilde_LB > utilde_UB):
            raise AssertionError('utilde bounds invalid after interval prop')
        if jnp.any(u_LB > u_UB):
            raise AssertionError('u bounds invalid after interval prop')
        if jnp.any(v_LB > v_UB):
            raise AssertionError('v bounds invalid after interval prop')

        utilde_LB, utilde_UB, u_LB, u_UB = BoundTightU(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)
        v_LB, v_UB = BoundTightV(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)

        if jnp.any(utilde_LB > utilde_UB):
            raise AssertionError('utilde bounds invalid after LP based bounds')
        if jnp.any(u_LB > u_UB):
            raise AssertionError('u bounds invalid after LP based bounds')
        if jnp.any(v_LB > v_UB):
            raise AssertionError('v bounds invalid after LP based bounds')

        result, time = ModelNextStep(k, A, c, t, utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB, momentum=momentum, beta_func=beta_func)
        log.info(result)

        Deltas.append(result)
        solvetimes.append(time)

        log.info(Deltas)
        log.info(solvetimes)

        Dk = jnp.sum(jnp.array(Deltas))
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

        df = pd.DataFrame(solvetimes)
        if cfg.momentum:
            df.to_csv(cfg.momentum_time_fname, index=False, header=False)
        else:
            df.to_csv(cfg.vanilla_time_fname, index=False, header=False)

        # plotting resids so far
        fig, ax = plt.subplots()
        ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
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
    if flow.u0.type == 'zero':
        # u0 = jnp.zeros(n)
        u0 = jnp.ones(n)  # TODO change back to zeros when done testing

    if flow.v0.type == 'zero':
        v0 = jnp.zeros(m)

    LP_run(cfg, jnp.asarray(A_block.todense()), c_tilde, t, u0, v0)


def run(cfg):
    if cfg.problem_type == 'flow':
        mincostflow_LP_run(cfg)
    else:
        random_LP_run(cfg)

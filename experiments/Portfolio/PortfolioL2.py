import copy
import logging

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB
from tqdm import trange

jnp.set_printoptions(precision=5)  # Print few decimal places
jnp.set_printoptions(suppress=True)  # Suppress scientific notation
jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14,
    "figure.figsize": (9, 4)})


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3
    absA = jnp.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_lower, Ax_upper


def proj_C(cfg, v):
    n = cfg.n
    m_plus_n = v.shape[0]
    if cfg.zprev.incl_upper_bound:
        return v.at[m_plus_n - 2 * n:].set(jax.nn.relu(v[m_plus_n - 2 * n:]))
    else:
        return v.at[m_plus_n - n:].set(jax.nn.relu(v[m_plus_n - n:]))


def BoundPreprocessing(cfg, k, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, c_l, c_u):
    m_plus_n = c_l.shape[0]
    utilde_A = jnp.block([[jnp.eye(m_plus_n), -jnp.eye(m_plus_n)]])
    utilde_A_tilde = jnp.linalg.solve(lhs_mat, utilde_A)
    utilde_rhs_l = jnp.hstack([s_LB[k-1], c_l])
    utilde_rhs_u = jnp.hstack([s_UB[k-1], c_u])

    utildek_LB, utildek_UB = interval_bound_prop(utilde_A_tilde, utilde_rhs_l, utilde_rhs_u)
    # log.info(utildek_LB)
    # log.info(utildek_UB)
    utilde_LB = utilde_LB.at[k].set(utildek_LB)
    utilde_UB = utilde_UB.at[k].set(utildek_UB)

    vA = jnp.block([[2 * jnp.eye(m_plus_n), -jnp.eye(m_plus_n)]])
    v_rhs_l = jnp.hstack([utilde_LB[k], s_LB[k-1]])
    v_rhs_u = jnp.hstack([utilde_UB[k], s_UB[k-1]])
    vk_LB, vk_UB = interval_bound_prop(vA, v_rhs_l, v_rhs_u)
    v_LB = v_LB.at[k].set(vk_LB)
    v_UB = v_UB.at[k].set(vk_UB)

    uk_LB = proj_C(cfg, vk_LB)
    uk_UB = proj_C(cfg, vk_UB)

    u_LB = u_LB.at[k].set(uk_LB)
    u_UB = u_UB.at[k].set(uk_UB)

    sA = jnp.block([[jnp.eye(m_plus_n), jnp.eye(m_plus_n), -jnp.eye(m_plus_n)]])
    s_rhs_l = jnp.hstack([s_LB[k-1], u_LB[k], utilde_LB[k]])
    s_rhs_u = jnp.hstack([s_UB[k-1], u_UB[k], utilde_UB[k]])
    sk_LB, sk_UB = interval_bound_prop(sA, s_rhs_l, s_rhs_u)

    s_LB = s_LB.at[k].set(sk_LB)
    s_UB = s_UB.at[k].set(sk_UB)

    return utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB


def sample_init_rad(cfg, P, A, b, s0, zprev_lower, zprev_upper, mu_l, mu_u, lambd):
    sample_idx = jnp.arange(cfg.samples.N)

    def s_sample(i):
        return s0

    def c_sample(i):
        key = jax.random.PRNGKey(cfg.samples.c_seed_offset + i)

        key, subkey = jax.random.split(key)
        mu = jax.random.uniform(subkey, shape=mu_l.shape, minval=mu_l, maxval=mu_u)
        # TODO add the if, start with box case only
        # return jax.random.uniform(key, shape=c_l.shape, minval=c_l, maxval=c_u)
        if cfg.zprev.l == 0:
            z_prev = sample_simplex(zprev_lower.shape[0], key)
        else:
            z_prev = jax.random.uniform(key, shape=zprev_lower.shape, minval=zprev_lower, maxval=zprev_upper)
        c = jnp.hstack([-(mu + 2 * lambd * z_prev), jnp.zeros(cfg.d), b])
        return c

    # s_samples = jax.vmap(s_sample)(sample_idx)
    c_samples = jax.vmap(c_sample)(sample_idx)
    distances = jnp.zeros(cfg.samples.init_dist_N)
    Am, An = A.shape

    for i in trange(cfg.samples.init_dist_N):
        z = cp.Variable(An)
        s = cp.Variable(Am)
        c_samp = c_samples[i]
        q = c_samp[:An]
        b = c_samp[An:]

        obj = cp.quad_form(z, P) + q.T @ z

        constraints = [
            A @ z + s == b,
            s[:cfg.d+1] == 0,
            s[cfg.d+1:] >= 0,
        ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve()

        cp_sol = np.hstack([z.value, constraints[0].dual_value])

        distances = distances.at[i].set(np.linalg.norm(cp_sol - s0, ord=cfg.C_norm))

    log.info(distances)
    return jnp.max(distances)


def compute_init_rad(cfg, P, A, b, s0, zprev_lower, zprev_upper, mu_l, mu_u, lambd):
    # TODO: add the l1 milp
    return sample_init_rad(cfg, P, A, b, s0, zprev_lower, zprev_upper, mu_l, mu_u, lambd)


def theory_bounds(k, R, s_LB, s_UB):
    log.info(f'-theory bound for k={k}-')

    const = 2 * R / np.sqrt(k)
    n = s_LB.shape[1]

    theory_tight_count = 0
    for i in range(n):
        if s_LB[k-1, i] - const > s_LB[k, i]:
            theory_tight_count += 1
            log.info(f'min, before: {s_LB[k, i]}, after {s_LB[k-1, i] - const}')
        s_LB = s_LB.at[k, i].set(max(s_LB[k, i], s_LB[k-1, i] - const))

        if s_UB[k-1, i] + const < s_UB[k, i]:
            theory_tight_count += 1
            log.info(f'max, before: {s_UB[k, i]}, after {s_UB[k-1, i] + const}')
        s_UB = s_UB.at[k, i].set(min(s_UB[k, i], s_UB[k-1, i] + const))

    return s_LB, s_UB, theory_tight_count / (2 * n)


def compute_lI(w, x, Lhat, Uhat, I, Icomp):
    if I.shape[0] == 0:
        return jnp.sum(jnp.multiply(w, Uhat))
    if Icomp.shape[0] == 0:
        return jnp.sum(jnp.multiply(w, Lhat))

    w_I = w[I]
    w_Icomp = w[Icomp]

    Lhat_I = Lhat[I]
    Uhat_I = Uhat[Icomp]

    return jnp.sum(jnp.multiply(w_I, Lhat_I)) + jnp.sum(jnp.multiply(w_Icomp, Uhat_I))


def compute_v_pos(wi, xi, Lhat, Uhat):
    idx = jnp.arange(wi.shape[0])
    # log.info(idx)

    filtered_idx = jnp.array([j for j in idx if wi[j] != 0 and jnp.abs(Uhat[j] - Lhat[j]) > 1e-7])
    # log.info(filtered_idx)

    def key_func(j):
        return (xi[j] - Lhat[j]) / (Uhat[j] - Lhat[j])

    keys = jnp.array([key_func(j) for j in filtered_idx])
    # log.info(keys)
    sorted_idx = jnp.argsort(keys)  # this is nondecreasing, should be the corrct one according to paper
    # sorted_idx = jnp.argsort(keys)[::-1]
    filtered_idx = filtered_idx[sorted_idx]

    # log.info(filtered_idx)

    I = jnp.array([])
    Icomp = set(range(wi.shape[0]))

    # log.info(Icomp)

    lI = compute_lI(wi, xi, Lhat, Uhat, I, jnp.array(list(Icomp)))
    # log.info(f'original lI: {lI}')
    if lI < 0:
        return None, None, None, None

    for h in filtered_idx:
        Itest = jnp.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        # log.info(Itest)
        # log.info(Icomp_test)

        lI_new = compute_lI(wi, xi, Lhat, Uhat, Itest.astype(jnp.integer), jnp.array(list(Icomp_test)))
        # log.info(lI_new)
        if lI_new < 0:
            Iint = I.astype(jnp.integer)
            # log.info(f'h={h}')
            # log.info(f'lI before and after: {lI}, {lI_new}')
            rhs = jnp.sum(jnp.multiply(wi[Iint], xi[Iint])) + lI / (Uhat[int(h)] - Lhat[int(h)]) * (xi[int(h)] - Lhat[int(h)])
            return Iint, rhs, lI, int(h)

        I = Itest
        Icomp = Icomp_test
        lI = lI_new
    else:
        return None, None, None, None


def add_pos_conv_cuts(cfg, k, i, s_LB, s_UB, utilde_LB, utilde_UB, rel_s, rel_utilde, rel_u):
    m_plus_n = rel_u.shape[0]
    L_hat = jnp.zeros(2 * m_plus_n)
    U_hat = jnp.zeros(2 * m_plus_n)

    # uk = Pi(2 utilde_k - skminus1)
    L_hat = L_hat.at[:m_plus_n].set(utilde_LB[k])
    L_hat = L_hat.at[m_plus_n:].set(s_UB[k-1])

    U_hat = U_hat.at[:m_plus_n].set(utilde_UB[k])
    U_hat = U_hat.at[m_plus_n:].set(s_LB[k-1])

    xi = jnp.hstack([rel_utilde, rel_s])
    wi = jnp.zeros(2 * m_plus_n)
    wi = wi.at[i].set(2)
    wi = wi.at[m_plus_n + i].set(-1)

    Iint, rhs, lI, h = compute_v_pos(wi, xi, L_hat, U_hat)

    if Iint is None:
        return None, None, None, None, None

    lhs = rel_u[i]
    if lhs > rhs + 1e-6:
        log.info('found a violated cut')
        log.info(f'with lI = {lI}')
        log.info(f'and I = {Iint}')
        log.info(f'h={h}')
        # exit(0)

    if lhs > rhs + 1e-6:
        return Iint, lI, h, L_hat, U_hat
    else:
        return None, None, None, None, None


def create_new_pos_constr(cfg, k, i, Iint, lI, h, s, utilde, u, Lhat, Uhat):

    m_plus_n = Lhat.shape[0]
    utilde_s_stack = gp.hstack([utilde[k], s[k-1]])
    wi = jnp.zeros(2 * m_plus_n)
    wi = wi.at[i].set(2)
    wi = wi.at[m_plus_n + i].set(-1)

    new_constr = 0
    for idx in Iint:
        new_constr += wi[idx] * (utilde_s_stack[idx] - Lhat[idx])
    new_constr += lI / (Uhat[h] - Lhat[h]) * (utilde_s_stack[h] - Lhat[h])

    return u[k][i] <= new_constr


def BuildRelaxedModel(cfg, K, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u):
    model = gp.Model()
    model.Params.OutputFlag = 0

    # jnp.linalg.solve(lhs_mat, utilde_A)
    # lhs_mat_inv = np.linalg.solve(lhs_mat, np.eye(lhs_mat.shape[0]))

    m, n = A.shape
    num_stocks, num_factors = cfg.n, cfg.d

    # c = model.addMVar(m + n, lb=c_l, ub=c_u)
    z_prev = model.addMVar(num_stocks, lb=zprev_lower, ub=zprev_upper)
    mu = model.addMVar(num_stocks, lb=mu_l, ub=mu_u)

    model.addConstr(gp.quicksum(z_prev) == 1)
    c = gp.hstack([-(mu + 2 * cfg.lambd * z_prev), np.zeros(num_factors), np.asarray(b)])

    utilde = model.addMVar((K+1, m+n), lb=utilde_LB[:K+1], ub=utilde_UB[:K+1])
    v = model.addMVar((K+1, m+n), lb=v_LB[:K+1], ub=v_UB[:K+1])
    u = model.addMVar((K+1, m+n), lb=u_LB[:K+1], ub=u_UB[:K+1])
    s = model.addMVar((K+1, m+n), lb=s_LB[:K+1], ub=s_UB[:K+1])

    for k in range(1, K+1):
        model.addConstr(lhs_mat @ utilde[k] == (s[k-1] - c))
        model.addConstr(s[k] == s[k-1] + u[k] - utilde[k])
        model.addConstr(v[k] == 2 * utilde[k] - s[k-1])

        # add relus
        for i in range(n + num_factors+1):
            model.addConstr(u[k][i] == v[k][i])

        for i in range(n + num_factors + 1, n + m):
            # log.info(i)
            if v_UB[k, i] <= 0:
                model.addConstr(u[k][i] == 0)
            elif v_LB[k, i] > 0:
                model.addConstr(u[k][i] == v[k][i])
            else:
                model.addConstr(u[k][i] >= v[k][i])
                model.addConstr(u[k][i] <= v_UB[k, i] / (v_UB[k, i] - v_LB[k, i]) * (v[k][i] - v_LB[k, i]))

    return model, utilde, v, s


def BoundTightUtilde(cfg, K, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u):
    log.info('bound tightening for utilde')
    m, n = A.shape
    model, utilde, _, _ = BuildRelaxedModel(cfg, K, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u)

    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        for i in range(m + n):
            model.setObjective(utilde[K, i], sense)
            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                log.info(f'bound tighting failed, GRB model status: {model.status}')
                log.info(f'(k, i) = {(K, i)}')

            if sense == GRB.MAXIMIZE:
                utilde_UB = utilde_UB.at[K, i].set(model.objVal)
            else:
                utilde_LB = utilde_LB.at[K, i].set(model.objVal)

            if utilde_LB[K, i] > utilde_UB[K, i]:
                raise ValueError('Infeasible bounds', sense, i, K, utilde_LB[K, i], utilde_UB[K, i])

    return utilde_LB, utilde_UB


def BoundTightVU(cfg, K, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u):
    log.info('bound tightening for v then u')
    m, n = A.shape
    model, _, v, _ = BuildRelaxedModel(cfg, K, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u)

    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        for i in range(m + n):
            model.setObjective(v[K, i], sense)
            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                log.info(f'bound tighting failed at v, GRB model status: {model.status}')
                log.info(f'(k, i) = {(K, i)}')
                exit(0)

            if sense == GRB.MAXIMIZE:
                v_UB = v_UB.at[K, i].set(model.objVal)
            else:
                v_LB = v_LB.at[K, i].set(model.objVal)

            if v_LB[K, i] > v_UB[K, i]:
                raise ValueError('Infeasible bounds', sense, i, K, v_LB[K, i], v_UB[K, i])

    u_UB = u_UB.at[K].set(proj_C(cfg, v_UB[K]))
    u_LB = u_LB.at[K].set(proj_C(cfg, v_LB[K]))

    return v_LB, v_UB, u_LB, u_UB


def BoundTightS(cfg, K, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u):
    log.info('bound tightening for s')
    m, n = A.shape
    model, _, _, s = BuildRelaxedModel(cfg, K, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u)

    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        for i in range(m + n):
            model.setObjective(s[K, i], sense)
            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                log.info(f'bound tighting failed at s, GRB model status: {model.status}')
                log.info(f'(k, i) = {(K, i)}')
                exit(0)

            if sense == GRB.MAXIMIZE:
                s_UB = s_UB.at[K, i].set(model.objVal)
            else:
                s_LB = s_LB.at[K, i].set(model.objVal)

            if s_LB[K, i] > s_UB[K, i]:
                raise ValueError('Infeasible bounds', sense, i, K, s_LB[K, i], s_UB[K, i])

    return s_LB, s_UB


def portfolio_verifier(cfg, D, A, b, s0, mu_l, mu_u):
    log.info(s0)

    def Init_model():
        model = gp.Model()
        model.setParam('TimeLimit', cfg.timelimit)
        model.setParam('MIPGap', cfg.mipgap)
        # model.setParam('MIPFocus', cfg.mipfocus)
        # model.setParam('OBBT', 0)
        # model.setParam('Cuts', 0)

        mu = model.addMVar(num_stocks, lb=mu_l, ub=mu_u)
        z_prev = model.addMVar(num_stocks, lb=zprev_lower, ub=zprev_upper)
        model.addConstr(gp.quicksum(z_prev) == 1)

        s[0] = model.addMVar(m + n, lb=s0, ub=s0)  # if non singleton, change here

        # q[:n] = -(mu + 2 * lambd / n)  # x_prev = 1/n

        c = gp.hstack([-(mu + 2 * lambd * z_prev), np.zeros(num_factors), np.asarray(b)])
        c_l = jnp.hstack([-(mu_u + 2 * lambd * zprev_upper), jnp.zeros(num_factors), b])
        c_u = jnp.hstack([-(mu_l + 2 * lambd * zprev_lower), jnp.zeros(num_factors), b])
        model.update()
        return model, c, mu, z_prev, c_l, c_u

    def ModelNextStep(model, k, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, obj_scaling=cfg.obj_scaling.default):
        obj_constraints = []

        # y[k] = model.addMVar(n, lb=y_LB[k], ub=y_UB[k])
        # z[k] = model.addMVar(n, lb=z_LB[k], ub=z_UB[k])
        # v[k] = model.addMVar(n, lb=v_LB[k], ub=v_UB[k])

        utilde[k] = model.addMVar(m+n, lb=utilde_LB[k], ub=utilde_UB[k])
        u[k] = model.addMVar(m+n, lb=u_LB[k], ub=u_UB[k])
        s[k] = model.addMVar(m+n, lb=s_LB[k], ub=s_UB[k])

        model.addConstr(lhs_mat @ utilde[k] == (s[k-1] - c))
        model.addConstr(s[k] == s[k-1] + u[k] - utilde[k])
        vk = 2 * utilde[k] - s[k-1]

        for i in range(n + num_factors+1):
            # log.info(i)
            model.addConstr(u[k][i] == vk[i])

        # log.info('-')
        for i in range(n + num_factors + 1, n + m):
            # log.info(i)
            if v_UB[k, i] <= 0:
                model.addConstr(u[k][i] == 0)
            elif v_LB[k, i] > 0:
                model.addConstr(u[k][i] == vk[i])
            else:
                w[k, i] = model.addVar(vtype=gp.GRB.BINARY)
                model.addConstr(u[k][i] >= vk[i])
                model.addConstr(u[k][i] <= v_UB[k, i] / (v_UB[k, i] - v_LB[k, i]) * (vk[i] - v_LB[k, i]))
                model.addConstr(u[k][i] <= vk[i] - v_LB[k, i] * (1 - w[k, i]))
                model.addConstr(u[k][i] <= v_UB[k, i] * w[k, i])

        # setting up for objective
        U = s_UB[k] - s_LB[k-1]
        L = s_LB[k] - s_UB[k-1]

        vobj = model.addMVar(m + n, vtype=gp.GRB.BINARY)
        up = model.addMVar(m + n, ub=jnp.abs(U))
        un = model.addMVar(m + n, ub=jnp.abs(L))

        if pnorm == 1 or pnorm == 'inf':
            obj_constraints.append(model.addConstr(up - un == s[k] - s[k-1]))

            # for i in range(m + n):
            #     # obj_constraints.append(up[i] <= np.abs(s_UB[k, i] - s_LB[k-1, i]) * vobj[i])
            #     # obj_constraints.append(un[i] <= np.abs(s_LB[k, i] - s_UB[k-1, i]) * (1 - vobj[i]))
            #     obj_constraints.append(model.addConstr(up[i] <= np.abs(s_UB[k, i] - s_LB[k-1, i]) * vobj[i]))
            #     obj_constraints.append(model.addConstr(un[i] <= np.abs(s_LB[k, i] - s_UB[k-1, i]) * (1 - vobj[i])))

            for i in range(m + n):
                if L[i] >= 0:
                    obj_constraints.append(model.addConstr(up[i] == s[k][i] - s[k-1][i]))
                    obj_constraints.append(model.addConstr(un[i] == 0))
                elif U[i] < 0:
                    obj_constraints.append(model.addConstr(un[i] == s[k-1][i] - s[k][i]))
                    obj_constraints.append(model.addConstr(up[i] == 0))
                else:
                    obj_constraints.append(model.addConstr(up[i] - un[i] == s[k][i] - s[k-1][i]))
                    obj_constraints.append(model.addConstr(up[i] <= jnp.abs(U[i]) * vobj[i]))
                    obj_constraints.append(model.addConstr(un[i] <= jnp.abs(L[i]) * (1-vobj[i])))

        if pnorm == 1:
            model.setObjective(1 / obj_scaling * gp.quicksum(up + un), GRB.MAXIMIZE)
        elif pnorm == 'inf':
            M = jnp.maximum(jnp.max(jnp.abs(U)), jnp.max(jnp.abs(L)))
            q = model.addVar(ub=M)
            gamma = model.addMVar(m + n, vtype=gp.GRB.BINARY)

            for i in range(m + n):
                obj_constraints.append(model.addConstr(q >= up[i] + un[i]))
                obj_constraints.append(model.addConstr(q <= up[i] + un[i] + M * (1 - gamma[i])))

            obj_constraints.append(model.addConstr(gp.quicksum(gamma) == 1))
            model.setObjective(1 / obj_scaling * q, gp.GRB.MAXIMIZE)

        model.update()
        if cfg.exact_conv_relax.use_in_l1_rel:
            rel_model = model.relax()
            rel_model.optimize()

            rel_s = np.array([])
            rel_utilde = np.array([])
            rel_u = np.array([])

            # uk = Pi(2 utilde_k - skminus1)
            for var in s[k-1]:
                rel_s = np.append(rel_s, rel_model.getVarByName(var.VarName.item()).X)

            for var in utilde[k]:
                rel_utilde = np.append(rel_utilde, rel_model.getVarByName(var.VarName.item()).X)

            for var in u[k]:
                rel_u = np.append(rel_u, rel_model.getVarByName(var.VarName.item()).X)

            log.info(rel_s)
            log.info(rel_utilde)
            log.info(rel_u)

            for i in range(n + num_factors + 1, n + m):
                log.info('--computing conv cuts--')
                if (k, i) in w:
                    log.info((k, i))
                    Iint, lI, h, L_hat, U_hat = add_pos_conv_cuts(cfg, k, i, s_LB, s_UB, utilde_LB, utilde_UB, rel_s, rel_utilde, rel_u)

                    if Iint is not None:
                        log.info(f'new lI constraint added with {(k, i)}')
                        model.addConstr(create_new_pos_constr(cfg, k, i, Iint, lI, h, s, utilde, u, L_hat, U_hat))

        model.update()
        model.optimize()

        for constr in obj_constraints:
            try:
                model.remove(constr)
            except gp.GurobiError:
                pass

        try:
            mipgap = model.MIPGap
        except AttributeError:
            mipgap = 0

        return model.objVal * obj_scaling, model.objBound * obj_scaling, mipgap, model.Runtime, mu.X, z_prev.X

    pnorm = cfg.pnorm
    K_max = cfg.K_max
    num_stocks, num_factors = cfg.n, cfg.d
    gamma, lambd = cfg.gamma, cfg.lambd
    m, n = A.shape

    zprev_lower = cfg.zprev.l * jnp.ones(num_stocks)
    zprev_upper = cfg.zprev.u * jnp.ones(num_stocks)

    P = 2 * jnp.block([
        [gamma * D + lambd * jnp.eye(num_stocks), jnp.zeros((num_stocks, num_factors))],
        [jnp.zeros((num_factors, num_stocks)), gamma * jnp.eye(num_factors)]
    ])

    M = jnp.block([
        [P, A.T],
        [-A, jnp.zeros((m, m))]
    ])

    lhs_mat = np.asarray(jnp.eye(m + n) + M)

    utilde = {}
    u = {}
    s = {}
    w = {}

    model, c, mu, z_prev, c_l, c_u = Init_model()
    log.info(P.shape)
    log.info(c.shape)
    log.info(M.shape)
    log.info(b)
    log.info(c_l)
    log.info(c_u)

    max_sample_resids = samples(cfg, lhs_mat, s0, c_l, c_u, zprev_lower, zprev_upper, mu_l, mu_u, lambd, b)
    log.info(f'max sample resids: {max_sample_resids}')

    utilde_LB = jnp.zeros((K_max + 1, m + n))
    utilde_UB = jnp.zeros((K_max + 1, m + n))

    v_LB = jnp.zeros((K_max + 1, m + n))
    v_UB = jnp.zeros((K_max + 1, m + n))

    u_LB = jnp.zeros((K_max + 1, m + n))
    u_UB = jnp.zeros((K_max + 1, m + n))

    s_LB = jnp.zeros((K_max + 1, m + n))
    s_UB = jnp.zeros((K_max + 1, m + n))

    s_LB = s_UB.at[0].set(s0)
    s_UB = s_UB.at[0].set(s0)

    Deltas = []
    Delta_bounds = []
    Delta_gaps = []
    solvetimes = []
    theory_tighter_fracs = []
    # c_out = jnp.zeros((K_max, m+n))

    #
    R = compute_init_rad(cfg, P, A, b, s0, zprev_lower, zprev_upper, mu_l, mu_u, lambd)
    log.info(f'init rad: {R}')
    obj_scaling = cfg.obj_scaling.default
    # exit(0)
    for k in range(1, K_max+1):
        log.info(f'---k={k}---')
        utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB = BoundPreprocessing(cfg, k, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, c_l, c_u)

        # TODO: insert assertion errors

        # TODO: add theory bound
        if cfg.theory_bounds:
            s_LB, s_UB, theory_tight_frac = theory_bounds(k, R, s_LB, s_UB)
            theory_tighter_fracs.append(theory_tight_frac)

        if cfg.opt_based_tightening:
            for _ in range(cfg.num_obbt_iter):
                utilde_LB, utilde_UB = BoundTightUtilde(cfg, k, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u)
                v_LB, v_UB, u_LB, u_UB = BoundTightVU(cfg, k, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u)
                s_LB, s_UB = BoundTightS(cfg, k, A, b, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, mu_l, mu_u)

        result, bound, opt_gap, time, mu_val, z_prev_val = ModelNextStep(model, k, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, obj_scaling=obj_scaling)

        log.info(result)
        log.info(mu_val)
        log.info(z_prev_val)

        Deltas.append(result)
        Delta_bounds.append(bound)
        Delta_gaps.append(opt_gap)
        solvetimes.append(time)

        if cfg.obj_scaling.val == 'adaptive':
            obj_scaling = result

        if cfg.postprocessing:
            Dk = jnp.sum(jnp.array(Delta_bounds))
            for i in range(n):
                s_LB = s_LB.at[k, i].set(max(s0[i] - Dk, s_LB[k, i]))
                s_UB = s_UB.at[k, i].set(min(s0[i] + Dk, s_UB[k, i]))
                s[k][i].LB = s_LB[k, i]
                s[k][i].UB = s_UB[k, i]

        model.update()

        df = pd.DataFrame(Deltas)  # remove the first column of zeros
        df.to_csv('resids.csv', index=False, header=False)

        df = pd.DataFrame(Delta_bounds)
        df.to_csv('resid_bounds.csv', index=False, header=False)

        df = pd.DataFrame(Delta_gaps)
        df.to_csv('resid_mip_gaps.csv', index=False, header=False)

        df = pd.DataFrame(solvetimes)
        df.to_csv('solvetimes.csv', index=False, header=False)

        if cfg.theory_bounds:
            df = pd.DataFrame(theory_tighter_fracs)
            df.to_csv('theory_tighter_fracs.csv', index=False, header=False)

        # plotting resids so far
        fig, ax = plt.subplots()
        ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
        ax.plot(range(1, len(Delta_bounds)+1), Delta_bounds, label='VP bounds', linewidth=5, alpha=0.3)
        ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

        ax.set_xlabel(r'$K$')
        ax.set_ylabel('Fixed-point residual')
        ax.set_yscale('log')
        ax.set_title(r'L2 Portfolio')

        ax.legend()
        plt.tight_layout()

        plt.savefig('resids.pdf')

        plt.clf()
        plt.cla()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(range(1, len(solvetimes)+1), solvetimes, label='VP')
        # ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

        ax.set_xlabel(r'$K$')
        ax.set_ylabel('Solvetime (s)')
        ax.set_yscale('log')
        ax.set_title(r'L1 Portfolio')

        ax.legend()

        plt.savefig('times.pdf')
        plt.clf()
        plt.cla()
        plt.close()

        log.info(f'max_sample_resids: {max_sample_resids}')
        log.info(f'Deltas: {Deltas}')
        log.info(f'times: {solvetimes}')
        log.info(f'theory tighter fracs: {theory_tighter_fracs}')


def avg_sol(cfg, D, A, b, mu):
    n, d = cfg.n, cfg.d
    z = cp.Variable(n + d)
    if cfg.zprev.incl_upper_bound:
        s = cp.Variable(2 * n + d + 1)
    else:
        s = cp.Variable(n + d + 1)
    gamma, lambd = cfg.gamma, cfg.lambd

    P = 2 * jnp.block([
        [gamma * D + lambd * jnp.eye(n), jnp.zeros((n, d))],
        [jnp.zeros((d, n)), gamma * jnp.eye(d)]
    ])

    q = np.zeros(n + d)
    q[:n] = -(mu + 2 * lambd / n)  # x_prev = 1/n

    obj = .5 * cp.quad_form(z, P) + q.T @ z
    constraints = [
        A @ z + s == b,
        s >= 0,
        s[:d+1] == 0,
    ]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    return jnp.hstack([z.value, constraints[0].dual_value])


def DR_alg(cfg, n, lu, piv, s0, c, K, pnorm='inf'):
    utildek_all = jnp.zeros((K+1, n))
    vk_all = jnp.zeros((K+1, n))
    uk_all = jnp.zeros((K+1, n))
    sk_all = jnp.zeros((K+1, n))
    resids = jnp.zeros(K+1)

    sk_all = sk_all.at[0].set(s0)

    # def proj()

    def body_fun(k, val):
        # sk_all, resids = val
        utildek_all, vk_all, uk_all, sk_all, resids = val
        sk = sk_all[k]

        # ykplus1 = At @ zk + Bt @ x
        # zkplus1 = soft_threshold(ykplus1, lambda_t)

        utilde_kplus1 = jsp.linalg.lu_solve((lu, piv), sk - c)
        vkplus1 = 2 * utilde_kplus1 - sk
        ukplus1 = proj_C(cfg, vkplus1)
        skplus1 = sk + ukplus1 - utilde_kplus1

        if pnorm == 'inf':
            resid = jnp.max(jnp.abs(skplus1 - sk))
        else:
            resid = jnp.linalg.norm(skplus1 - sk, ord=pnorm)

        utildek_all = utildek_all.at[k+1].set(utilde_kplus1)
        vk_all = vk_all.at[k+1].set(vkplus1)
        uk_all = uk_all.at[k+1].set(ukplus1)
        sk_all = sk_all.at[k+1].set(skplus1)
        resids = resids.at[k+1].set(resid)
        # return (sk_all, resids)
        return (utildek_all, vk_all, uk_all, sk_all, resids)

    # return jax.lax.fori_loop(0, K, body_fun, (sk_all, resids))
    return jax.lax.fori_loop(0, K, body_fun, (utildek_all, vk_all, uk_all, sk_all, resids))


def sample_simplex(n, key):
    intervals = jax.random.uniform(key, shape=(n-1,))
    intervals = jnp.sort(intervals)
    all_vals = jnp.zeros(n+1)
    all_vals = all_vals.at[1:n].set(intervals)
    all_vals = all_vals.at[n].set(1)
    # log.info(all_vals)
    x = np.zeros(n)

    def body_fun(i, val):
        return val.at[i].set(all_vals[i+1] - all_vals[i])

    return jax.lax.fori_loop(0, n, body_fun, x)


def samples(cfg, lhs_mat, s0, c_l, c_u, zprev_lower, zprev_upper, mu_l, mu_u, lambd, b):
    lu, piv, _ = jax.lax.linalg.lu(lhs_mat)  # usage: sol = jsp.linalg.lu_solve((lu, piv), rhs)

    sample_idx = jnp.arange(cfg.samples.N)

    def s_sample(i):
        return s0

    # c = gp.hstack([-(mu + 2 * lambd * z_prev), np.zeros(num_factors), np.asarray(b)])

    def c_sample(i):
        key = jax.random.PRNGKey(cfg.samples.c_seed_offset + i)

        key, subkey = jax.random.split(key)
        mu = jax.random.uniform(subkey, shape=mu_l.shape, minval=mu_l, maxval=mu_u)
        # TODO add the if, start with box case only
        # return jax.random.uniform(key, shape=c_l.shape, minval=c_l, maxval=c_u)
        if cfg.zprev.l == 0:
            z_prev = sample_simplex(zprev_lower.shape[0], key)
        else:
            z_prev = jax.random.uniform(key, shape=zprev_lower.shape, minval=zprev_lower, maxval=zprev_upper)
        c = jnp.hstack([-(mu + 2 * lambd * z_prev), jnp.zeros(cfg.d), b])
        return c

    s_samples = jax.vmap(s_sample)(sample_idx)
    c_samples = jax.vmap(c_sample)(sample_idx)

    def dr_resids(i):
        return DR_alg(cfg, s0.shape[0], lu, piv, s_samples[i], c_samples[i], cfg.K_max, pnorm=cfg.pnorm)

    _, _, _, _, sample_resids = jax.vmap(dr_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def portfolio_l2(cfg):
    log.info(cfg)
    n, d = cfg.n, cfg.d

    key = jax.random.PRNGKey(cfg.data_rng_key)

    key, subkey = jax.random.split(key)
    F = jax.random.normal(subkey, shape=(n, d))

    key, subkey = jax.random.split(key)
    F_mask = jax.random.bernoulli(subkey, p=cfg.F_mask_prob, shape=(n, d)).astype(jnp.float64)

    F = jnp.multiply(F, F_mask)
    log.info(F)

    key, subkey = jax.random.split(key)
    Ddiag = jax.random.uniform(subkey, shape=(n, ), maxval = 1/jnp.sqrt(d))

    D = jnp.diag(Ddiag)
    log.info(D)

    if cfg.zprev.incl_upper_bound:
        A = np.block([
            [F.T, -jnp.eye(d)],
            [jnp.ones((1, n)), jnp.zeros((1, d))],
            [-jnp.eye(n), jnp.zeros((n, d))],
            [jnp.eye(n), jnp.zeros((n, d))]
        ])
        b = jnp.hstack([jnp.zeros(d), 1, jnp.zeros(n), cfg.zprev.u * jnp.ones(n)])
    else:
        A = np.block([
            [F.T, -jnp.eye(d)],
            [jnp.ones((1, n)), jnp.zeros((1, d))],
            [-jnp.eye(n), jnp.zeros((n, d))]
        ])
        b = jnp.hstack([jnp.zeros(d), 1, jnp.zeros(n)])

    log.info(A.shape)
    log.info(b.shape)

    mu_l = cfg.mu.l * jnp.ones(n)
    mu_u = cfg.mu.u * jnp.ones(n)

    # model.addConstr(z_prev == np.array([0, 0, 0, 1]))
    # model.addConstr(mu == np.array([-.25, -.25, .25, -.25]))

    # mu_l = np.array([-.15, -.15, .15, -.15])
    # mu_u = np.array([-.15, -.15, .15, -.15])

    if cfg.z0.type == 'avg_sol':
        key, subkey = jax.random.split(key)
        mu_sample = jax.random.uniform(subkey, shape=(n,), minval=mu_l, maxval=mu_u)
        s0 = avg_sol(cfg, D, A, b, mu_sample)
    elif cfg.z0.type == 'zero':
        s0 = jnp.zeros(2 * n + 2 * d + 1)

    # y0 = F.T @ z0

    # portfolio_verifier(cfg, D, A, b, jnp.hstack([z0, y0]), mu_l, mu_u)
    portfolio_verifier(cfg, D, A, b, s0, mu_l, mu_u)


def run(cfg):
    portfolio_l2(cfg)

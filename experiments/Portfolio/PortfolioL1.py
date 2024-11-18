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

# from scipy.stats import dirichlet

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
    offset = 3 * n
    m_plus_n = v.shape[0]
    return v.at[m_plus_n - offset:].set(jax.nn.relu(v[m_plus_n - offset:]))


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


def samples(cfg, lhs_mat, s0, alpha_l, alpha_u, mu_l, mu_u, lambd_l, lambd_u, zprev_lower, zprev_upper):
    lu, piv, _ = jax.lax.linalg.lu(lhs_mat)  # usage: sol = jsp.linalg.lu_solve((lu, piv), rhs)

    sample_idx = jnp.arange(cfg.samples.N)

    def s_sample(i):
        return s0

    def c_sample(i):
        key = jax.random.PRNGKey(cfg.samples.c_seed_offset + i)
        key, subkey = jax.random.split(key)
        # TODO add the if, start with box case only
        # return jax.random.uniform(key, shape=c_l.shape, minval=c_l, maxval=c_u)
        alpha = jax.random.uniform(subkey, minval=alpha_l, maxval=alpha_u)

        key, subkey = jax.random.split(key)
        mu = jax.random.uniform(subkey, shape=mu_l.shape, minval=mu_l, maxval=mu_u)

        key, subkey = jax.random.split(key)
        lambd = jax.random.uniform(subkey, minval=lambd_l, maxval=lambd_u)

        q = jnp.hstack([-alpha* mu, jnp.zeros(cfg.d), alpha * lambd * jnp.ones(cfg.n)])

        # q = jax.random.uniform(subkey, shape=q_l.shape, minval=q_l, maxval=q_u)
        # z_prev = jax.random.uniform(key, shape=zprev_lower.shape, minval=zprev_lower, maxval=zprev_upper)

        if cfg.zprev.l == 0:
            z_prev = sample_simplex(zprev_lower.shape[0], key)
        else:
            z_prev = jax.random.uniform(key, shape=zprev_lower.shape, minval=zprev_lower, maxval=zprev_upper)

        if cfg.zprev.incl_upper_bound:
            b = jnp.hstack([jnp.zeros(cfg.d), 1, z_prev, -z_prev, jnp.zeros(cfg.n), cfg.zprev.incl_upper_bound * jnp.ones(cfg.n)])
        else:
            b = jnp.hstack([jnp.zeros(cfg.d), 1, z_prev, -z_prev, jnp.zeros(cfg.n)])
        return jnp.hstack([q, b])

    s_samples = jax.vmap(s_sample)(sample_idx)
    c_samples = jax.vmap(c_sample)(sample_idx)

    # log.info(c_samples[:, 2 * cfg.n + 2 * cfg.d + 1: 4 * cfg.n + 2 * cfg.d + 1])
    # log.info(c_samples[0] )
    # exit(0)

    def dr_resids(i):
        return DR_alg(cfg, s0.shape[0], lu, piv, s_samples[i], c_samples[i], cfg.K_max, pnorm=cfg.pnorm)

    _, _, _, _, sample_resids = jax.vmap(dr_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def compute_sample_init_rad(cfg, P, A, s0, alpha_l, alpha_u, mu_l, mu_u, lambd_l, lambd_u, zprev_lower, zprev_upper):
    sample_idx = jnp.arange(cfg.samples.N)

    def s_sample(i):
        return s0

    # TODO: remove the duplicate code here and in samples
    def c_sample(i):
        key = jax.random.PRNGKey(cfg.samples.c_seed_offset + i)
        key, subkey = jax.random.split(key)
        # TODO add the if, start with box case only
        # return jax.random.uniform(key, shape=c_l.shape, minval=c_l, maxval=c_u)
        alpha = jax.random.uniform(subkey, minval=alpha_l, maxval=alpha_u)

        key, subkey = jax.random.split(key)
        mu = jax.random.uniform(subkey, shape=mu_l.shape, minval=mu_l, maxval=mu_u)

        key, subkey = jax.random.split(key)
        lambd = jax.random.uniform(subkey, minval=lambd_l, maxval=lambd_u)

        q = jnp.hstack([-alpha* mu, jnp.zeros(cfg.d), alpha * lambd * jnp.ones(cfg.n)])

        if cfg.zprev.l == 0:
            z_prev = sample_simplex(zprev_lower.shape[0], key)
        else:
            z_prev = jax.random.uniform(key, shape=zprev_lower.shape, minval=zprev_lower, maxval=zprev_upper)

        if cfg.zprev.incl_upper_bound:
            b = jnp.hstack([jnp.zeros(cfg.d), 1, z_prev, -z_prev, jnp.zeros(cfg.n), cfg.zprev.incl_upper_bound * jnp.ones(cfg.n)])
        else:
            b = jnp.hstack([jnp.zeros(cfg.d), 1, z_prev, -z_prev, jnp.zeros(cfg.n)])
        return jnp.hstack([q, b])

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


def compute_init_rad(cfg, P, A, s0, alpha_l, alpha_u, mu_l, mu_u, lambd_l, lambd_u, zprev_lower, zprev_upper):
    # TODO: add the l1 mip
    return compute_sample_init_rad(cfg, P, A, s0, alpha_l, alpha_u, mu_l, mu_u, lambd_l, lambd_u, zprev_lower, zprev_upper)


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


def BuildRelaxedModel(cfg, K, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u):

    model = gp.Model()
    model.Params.OutputFlag = 0

    # jnp.linalg.solve(lhs_mat, utilde_A)
    # lhs_mat_inv = np.linalg.solve(lhs_mat, np.eye(lhs_mat.shape[0]))

    m, n = A.shape
    num_stocks, num_factors = cfg.n, cfg.d

    alpha_mu = model.addMVar(num_stocks, lb=alpha_mu_l, ub=alpha_mu_u)
    alpha_lambd = model.addMVar(1, lb=alpha_lambd_l, ub=alpha_lambd_u)
    one_var = model.addMVar(1, lb=1, ub=1)  # needed for gp.hstack

    z_prev = model.addMVar(num_stocks, lb=zprev_lower, ub=zprev_upper)
    model.addConstr(gp.quicksum(z_prev) == 1)

    q = gp.hstack([-alpha_mu, np.zeros(num_factors), alpha_lambd * np.ones(num_stocks)])
    if cfg.zprev.incl_upper_bound:
        b = gp.hstack([np.zeros(num_factors), one_var, z_prev, -z_prev, np.zeros(num_stocks), cfg.zprev.u * np.ones(num_stocks)])
    else:
        b = gp.hstack([np.zeros(num_factors), one_var, z_prev, -z_prev, np.zeros(num_stocks)])
    c = gp.hstack([q, b])

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


def BoundTightUtilde(cfg, K, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u):
    log.info('bound tightening for utilde')
    m, n = A.shape
    model, utilde, _, _ = BuildRelaxedModel(cfg, K, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u)

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

            if utilde_LB[K, i] > utilde_UB[K, i] + 1e-9:
                log.info(utilde_LB[K, i] - utilde_UB[K, i])
                raise ValueError('Infeasible bounds', sense, i, K, utilde_LB[K, i], utilde_UB[K, i])

    return utilde_LB, utilde_UB


def BoundTightVU(cfg, K, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u):
    log.info('bound tightening for v then u')
    m, n = A.shape
    model, _, v, _ = BuildRelaxedModel(cfg, K, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u)

    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        for i in range(m + n):
            model.setObjective(v[K, i], sense)
            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                log.info(f'bound tighting failed, GRB model status: {model.status}')
                log.info(f'(k, i) = {(K, i)}')

            if sense == GRB.MAXIMIZE:
                v_UB = v_UB.at[K, i].set(model.objVal)
            else:
                v_LB = v_LB.at[K, i].set(model.objVal)

            if v_LB[K, i] > v_UB[K, i]:
                raise ValueError('Infeasible bounds', sense, i, K, v_LB[K, i], v_UB[K, i])

    u_UB = u_UB.at[K].set(proj_C(cfg, v_UB[K]))
    u_LB = u_LB.at[K].set(proj_C(cfg, v_LB[K]))

    return v_LB, v_UB, u_LB, u_UB


def BoundTightS(cfg, K, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u):
    log.info('bound tightening for s')
    m, n = A.shape
    model, _, _, s = BuildRelaxedModel(cfg, K, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u)

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


def portfolio_verifier(cfg, D, A, s0):

    def Init_model():
        model = gp.Model()
        model.setParam('TimeLimit', cfg.timelimit)
        model.setParam('MIPGap', cfg.mipgap)
        # model.setParam('MIPFocus', cfg.mipfocus)
        # model.setParam('OBBT', 0)
        # model.setParam('Cuts', 0)

        # mu = model.addMVar(num_stocks, lb=mu_l, ub=mu_u)
        alpha_mu = model.addMVar(num_stocks, lb=alpha_mu_l, ub=alpha_mu_u)
        alpha_lambd = model.addMVar(1, lb=alpha_lambd_l, ub=alpha_lambd_u)
        one_var = model.addMVar(1, lb=1, ub=1)  # needed for gp.hstack

        z_prev = model.addMVar(num_stocks, lb=zprev_lower, ub=zprev_upper)
        model.addConstr(gp.quicksum(z_prev) == 1)

        s[0] = model.addMVar(m + n, lb=s0, ub=s0)  # if non singleton, change here
        q = gp.hstack([-alpha_mu, np.zeros(num_factors), alpha_lambd * np.ones(num_stocks)])
        # b = gp.hstack([np.zeros(num_factors), one_var, z_prev, -z_prev, np.zeros(num_stocks)])

        q_l = np.hstack([-alpha_mu_u, np.zeros(num_factors), alpha_lambd_l * np.ones(num_stocks)])
        q_u = np.hstack([-alpha_mu_l, np.zeros(num_factors), alpha_lambd_u * np.ones(num_stocks)])

        if cfg.zprev.incl_upper_bound:
            b = gp.hstack([np.zeros(num_factors), one_var, z_prev, -z_prev, np.zeros(num_stocks), cfg.zprev.u * np.ones(num_stocks)])
            b_l = np.hstack([np.zeros(num_factors), 1, zprev_lower, -zprev_upper, np.zeros(num_stocks), cfg.zprev.u * np.ones(num_stocks)])
            b_u = np.hstack([np.zeros(num_factors), 1, zprev_upper, -zprev_lower, np.zeros(num_stocks), cfg.zprev.u * np.ones(num_stocks)])
        else:
            b = gp.hstack([np.zeros(num_factors), one_var, z_prev, -z_prev, np.zeros(num_stocks)])
            b_l = np.hstack([np.zeros(num_factors), 1, zprev_lower, -zprev_upper, np.zeros(num_stocks)])
            b_u = np.hstack([np.zeros(num_factors), 1, zprev_upper, -zprev_lower, np.zeros(num_stocks)])

        c = gp.hstack([q, b])
        c_l = np.hstack([q_l, b_l])
        c_u = np.hstack([q_u, b_u])
        model.update()
        return model, c, c_l, c_u, q_l, q_u, b_l, b_u

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

            for i in range(m + n):
                obj_constraints.append(up[i] <= np.abs(s_UB[k, i] - s_LB[k-1, i]) * vobj[i])
                obj_constraints.append(un[i] <= np.abs(s_LB[k, i] - s_UB[k-1, i]) * (1 - vobj[i]))

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

        return model.objVal * obj_scaling, model.objBound * obj_scaling, mipgap, model.Runtime

    pnorm = cfg.pnorm
    K_max = cfg.K_max
    num_stocks, num_factors = cfg.n, cfg.d
    # gamma, lambd = cfg.gamma, cfg.lambd
    gamma_l, gamma_u = cfg.gamma_l, cfg.gamma_u
    alpha_l, alpha_u = 1 / gamma_u, 1 / gamma_l
    lambd_l, lambd_u = cfg.lambd_l, cfg.lambd_u
    m, n = A.shape

    zprev_lower = cfg.zprev.l * np.ones(num_stocks)
    zprev_upper = cfg.zprev.u * np.ones(num_stocks)
    # zprev_lower = np.array([1, 0, 0, 0, 0])
    # zprev_upper = np.array([1, 0, 0, 0, 0])

    mu_l = cfg.mu.l * np.ones(num_stocks)
    mu_u = cfg.mu.u * np.ones(num_stocks)

    if cfg.mu.l >= 0 or cfg.mu.u <= 0:
        log.info('note bound prop assumed mu_l < 0 and mu_u > 0')
        log.info('need to implement otherwise')
        exit(0)

    alpha_mu_l = alpha_u * mu_l
    alpha_mu_u = alpha_u * mu_u

    alpha_lambd_l = alpha_l * lambd_l
    alpha_lambd_u = alpha_u * lambd_u

    P = 2 * np.block([
        [D, np.zeros((num_stocks, num_factors)), np.zeros((num_stocks, num_stocks))],
        [np.zeros((num_factors, num_stocks)), np.eye(num_factors), np.zeros((num_factors, num_stocks))],
        [np.zeros((num_stocks, num_stocks)), np.zeros((num_stocks, num_factors)), np.zeros((num_stocks, num_stocks))]
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

    model, c, c_l, c_u, q_l, q_u, b_l, b_u = Init_model()

    log.info(P.shape)
    log.info(A.shape)
    log.info(c.shape)
    log.info(M.shape)
    log.info(c_l.shape)
    log.info(c_u)

    max_sample_resids = samples(cfg, lhs_mat, s0, alpha_l, alpha_u, mu_l, mu_u, lambd_l, lambd_u, zprev_lower, zprev_upper)
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

    obj_scaling = cfg.obj_scaling.default
    R = compute_init_rad(cfg, P, A, s0, alpha_l, alpha_u, mu_l, mu_u, lambd_l, lambd_u, zprev_lower, zprev_upper)
    log.info(f'initial radius: {R}')

    for k in range(1, K_max+1):
        log.info(f'---k={k}---')
        utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB = BoundPreprocessing(cfg, k, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, c_l, c_u)

        if cfg.theory_bounds:
            s_LB, s_UB, theory_tight_frac = theory_bounds(k, R, s_LB, s_UB)
            theory_tighter_fracs.append(theory_tight_frac)

        if cfg.opt_based_tightening:
            for _ in range(cfg.num_obbt_iter):
                utilde_LB, utilde_UB = BoundTightUtilde(cfg, k, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u)
                v_LB, v_UB, u_LB, u_UB = BoundTightVU(cfg, k, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u)
                s_LB, s_UB = BoundTightS(cfg, k, A, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, zprev_lower, zprev_upper, alpha_mu_l, alpha_mu_u, alpha_lambd_l, alpha_lambd_u)

        result, bound, opt_gap, time = ModelNextStep(model, k, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, obj_scaling=obj_scaling)

        Deltas.append(result)
        Delta_bounds.append(bound)
        Delta_gaps.append(opt_gap)
        solvetimes.append(time)

        if cfg.obj_scaling.val == 'adaptive':
            obj_scaling = result

        log.info(f'max samples: {max_sample_resids}')
        log.info(f'Deltas: {Deltas}')
        log.info(f'solvetimes: {solvetimes}')
        log.info(theory_tighter_fracs)

        if cfg.postprocessing:
            log.info('--postprocessing--')
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
        ax.set_title(r'L1 Portfolio')

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

        # x_out_plot = x_out.T

        # plt.imshow(x_out_plot, cmap='viridis')
        # plt.colorbar()

        # plt.xlabel(r'$K$')
        # plt.savefig('x_heatmap.pdf')

        # df = pd.DataFrame(x_out_plot)
        # df.to_csv('x_heatmap.csv', index=False, header=False)

        # plt.clf()
        # plt.cla()
        # plt.close()

        log.info(f'max_sample_resids: {max_sample_resids}')
        log.info(f'Deltas: {Deltas}')
        log.info(f'times: {solvetimes}')
        log.info(f'theory tighter fracs: {theory_tighter_fracs}')


def avg_sol(cfg, D, A, b, mu_sample):
    n, d = cfg.n, cfg.d
    z = cp.Variable(2 * n + d)
    if cfg.zprev.incl_upper_bound:
        s = cp.Variable(4 * n + d + 1)
    else:
        s = cp.Variable(3 * n + d + 1)
    gamma, lambd = cfg.gamma, cfg.lambd
    alpha = 1 / gamma

    P = 2 * np.block([
        [D, np.zeros((n, d)), np.zeros((n, n))],
        [np.zeros((d, n)), np.eye(d), np.zeros((d, n))],
        [np.zeros((n, n)), np.zeros((n, d)), np.zeros((n, n))]
    ])

    q = jnp.hstack([-alpha * mu_sample, jnp.zeros(d), alpha * lambd * jnp.ones(n)])

    log.info(P.shape)
    log.info(q.shape)

    obj = .5 * cp.quad_form(z, P) + q.T @ z
    constraints = [
        A @ z + s == b,
        s[:d+1] == 0,
        s[d+1:] >= 0,
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    return jnp.hstack([z.value, constraints[0].dual_value])


def portfolio_l1(cfg):
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
        A = jnp.block([
            [F.T, -jnp.eye(d), jnp.zeros((d, n))],
            [jnp.ones((1, n)), jnp.zeros((1, d)), jnp.zeros((1, n))],
            [jnp.eye(n), jnp.zeros((n, d)), -jnp.eye(n)],
            [-jnp.eye(n), jnp.zeros((n, d)), -jnp.eye(n)],
            [-jnp.eye(n), jnp.zeros((n, d)), jnp.zeros((n, n))],
            [jnp.eye(n), jnp.zeros((n, d)), jnp.zeros((n, n))]
        ])
    else:
        A = jnp.block([
            [F.T, -jnp.eye(d), jnp.zeros((d, n))],
            [jnp.ones((1, n)), jnp.zeros((1, d)), jnp.zeros((1, n))],
            [jnp.eye(n), jnp.zeros((n, d)), -jnp.eye(n)],
            [-jnp.eye(n), jnp.zeros((n, d)), -jnp.eye(n)],
            [-jnp.eye(n), jnp.zeros((n, d)), jnp.zeros((n, n))]
        ])

    # log.info(A.shape)
    # log.info(3 * n + d + 1)

    mu_l = cfg.mu.l * jnp.ones(n)
    mu_u = cfg.mu.u * jnp.ones(n)

    z_prev = 1 / n

    if cfg.zprev.incl_upper_bound:
        b_avg = jnp.hstack([jnp.zeros(d), 1, z_prev * jnp.ones(n), -z_prev * jnp.ones(n), jnp.zeros(n), cfg.zprev.u * jnp.ones(n)])
    else:
        b_avg = jnp.hstack([jnp.zeros(d), 1, z_prev * jnp.ones(n), -z_prev * jnp.ones(n), jnp.zeros(n)])

    if cfg.z0.type == 'avg_sol':
        key, subkey = jax.random.split(key)
        mu_sample = jax.random.uniform(subkey, shape=(n,), minval=mu_l, maxval=mu_u)
        s0 = avg_sol(cfg, D, A, b_avg, mu_sample)
    elif cfg.z0.type == 'zero':
        # s0 = jnp.zeros(3 * n + 2 * d + 1)
        Am, An = A.shape
        s0 = jnp.zeros(Am + An)

    log.info(f's0: {s0}')

    portfolio_verifier(cfg, D, A, s0)


def run(cfg):
    portfolio_l1(cfg)

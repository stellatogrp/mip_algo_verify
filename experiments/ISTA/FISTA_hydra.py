import logging

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB

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


def get_Dk_Ek(k, n, gamma):
    Dk = (1 + gamma[k]) * np.eye(n)
    Ek = -gamma[k] * np.eye(n)
    return np.asarray(Dk), np.asarray(Ek)


def BoundPreprocessing(k, At, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, Btx_LB, Btx_UB, gamma, lambda_t):
    Atvk_UB, Atvk_LB = interval_bound_prop(At, v_LB[k-1], v_UB[k-1])
    yk_UB = Atvk_UB + Btx_UB
    yk_LB = Atvk_LB + Btx_LB

    if jnp.any(yk_UB < yk_LB):
        raise AssertionError('basic y bound prop failed')

    y_LB = y_LB.at[k].set(yk_LB)
    y_UB = y_UB.at[k].set(yk_UB)
    z_LB = z_LB.at[k].set(soft_threshold(yk_LB, lambda_t))
    z_UB = z_UB.at[k].set(soft_threshold(yk_UB, lambda_t))

    Dk, Ek = get_Dk_Ek(k, At.shape[0], gamma)

    Dkzk_UB, Dkzk_LB = interval_bound_prop(Dk, z_LB[k], z_UB[k])
    Ekzkminus1_UB, Ekzkminus1_LB = interval_bound_prop(Ek, z_LB[k-1], z_UB[k-1])

    vk_UB = Dkzk_UB + Ekzkminus1_UB
    vk_LB = Dkzk_LB + Ekzkminus1_LB

    if jnp.any(vk_UB < vk_LB):
        raise AssertionError('basic v bound prop failed')

    v_LB = v_LB.at[k].set(vk_LB)
    v_UB = v_UB.at[k].set(vk_UB)

    return y_LB, y_UB, z_LB, z_UB, v_LB, v_UB


def BuildRelaxedModel(K, At, Bt, gamma, lambda_t, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, x_l, x_u):
    n, m = Bt.shape

    At = np.asarray(At)
    Bt = np.asarray(Bt)

    model = gp.Model()
    model.Params.OutputFlag = 0

    x = model.addMVar(m, lb=x_l, ub=x_u)

    # NOTE: we do NOT have bounds on zk or vk yet, so only bound up to zk-1 and prop forward to yk

    z = model.addMVar((K+1, n), lb=z_LB[:K+1], ub=z_UB[:K+1])
    y = model.addMVar((K+1, n), lb=y_LB[:K+1], ub=y_UB[:K+1])
    v = model.addMVar((K+1, n), lb=v_LB[:K+1], ub=v_UB[:K+1])

    for k in range(1, K+1):
        Dk, Ek = get_Dk_Ek(k, n, gamma)
        model.addConstr(v[k] == Dk @ z[k] + Ek @ z[k-1])

    for k in range(1, K+1):
        model.addConstr(y[k] == At @ z[k-1] + Bt @ x)

    for k in range(1, K+1):
        for i in range(n):
            if y_LB[k, i] >= lambda_t:
                model.addConstr(z[k, i] == y[k, i] - lambda_t)

            elif y_UB[k, i] <= -lambda_t:
                model.addConstr(z[k, i] == y[k, i] + lambda_t)

            elif y_LB[k, i] >= -lambda_t and y_UB[k, i] <= lambda_t:
                model.addConstr(z[k, i] == 0.0)

            elif y_LB[k, i] < -lambda_t and y_UB[k, i] > lambda_t:
                model.addConstr(z[k, i] >= y[k, i] - lambda_t)
                model.addConstr(z[k, i] <= y[k, i] + lambda_t)

                model.addConstr(z[k, i] <= z_UB[k, i]/(y_UB[k, i] + lambda_t)*(y[k, i] + lambda_t))
                model.addConstr(z[k, i] >= z_LB[k, i]/(y_LB[k, i] - lambda_t)*(y[k, i] - lambda_t))

            elif -lambda_t <= y_LB[k, i] <= lambda_t and y_UB[k, i] > lambda_t:
                model.addConstr(z[k, i] >= 0)
                model.addConstr(z[k, i] <= z_UB[k, i]/(y_UB[k, i] - y_LB[k, i])*(y[k, i] - y_LB[k, i]))
                model.addConstr(z[k, i] >= y[k, i] - lambda_t)

            elif -lambda_t <= y_UB[k, i] <= lambda_t and y_LB[k, i] < -lambda_t:
                model.addConstr(z[k, i] <= 0)
                model.addConstr(z[k, i] >= z_LB[k, i]/(y_LB[k, i] - y_UB[k, i])*(y[k, i] - y_UB[k, i]))
                model.addConstr(z[k, i] <= y[k, i] + lambda_t)
            else:
                raise RuntimeError('Unreachable code', y_LB[k, i], y_UB[k, i], lambda_t)

    model.update()
    return model, y, v


def BoundTightY(k, At, Bt, gamma, lambda_t, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, x_LB, x_UB):
    model, y, _ = BuildRelaxedModel(k, At, Bt, gamma, lambda_t, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, x_LB, x_UB)
    n = At.shape[0]

    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        for i in range(n):
            model.setObjective(y[k, i], sense)
            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print('bound tighting failed, GRB model status:', model.status)
                return None

            if sense == GRB.MAXIMIZE:
                y_UB = y_UB.at[k, i].set(model.objVal)
            else:
                y_LB = y_LB.at[k, i].set(model.objVal)

            if y_LB[k, i] > y_UB[k, i]:
                raise ValueError('Infeasible bounds', sense, i, k, y_LB[k, i], y_UB[k, i])

    z_UB = z_UB.at[k].set(soft_threshold(y_UB[k], lambda_t))
    z_LB = z_LB.at[k].set(soft_threshold(y_LB[k], lambda_t))

    return y_LB, y_UB, z_LB, z_UB


def BoundTightV(k, At, Bt, gamma, lambda_t, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, x_LB, x_UB):
    model, _, v = BuildRelaxedModel(k, At, Bt, gamma, lambda_t, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, x_LB, x_UB)
    n = At.shape[0]

    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        for i in range(n):
            model.setObjective(v[k, i], sense)
            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print('bound tighting failed, GRB model status:', model.status)
                return None

            if sense == GRB.MAXIMIZE:
                v_UB = v_UB.at[k, i].set(model.objVal)
            else:
                v_LB = v_LB.at[k, i].set(model.objVal)

            if v_LB[k, i] > v_UB[k, i]:
                raise ValueError('Infeasible bounds', sense, i, k, v_LB[k, i], v_UB[k, i])

    return v_LB, v_UB


def FISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u):

    def Init_model():
        model = gp.Model()
        model.setParam('TimeLimit', cfg.timelimit)
        model.setParam('MIPGap', cfg.mipgap)

        x = model.addMVar(m, lb=x_l, ub=x_u)
        z[0] = model.addMVar(n, lb=c_z, ub=c_z)  # if non singleton, change here
        v[0] = model.addMVar(n, lb=c_z, ub=c_z)

        model.update()
        return model, x

    def ModelNextStep(model, k, At, Bt, lambda_t, c_z, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, obj_scaling=cfg.obj_scaling.default):
        # obj_constraints = []
        pass

    max_sample_resids = samples(cfg, A, lambd, t, c_z, x_l, x_u)
    log.info(f'max sample resids: {max_sample_resids}')

    # pnorm = cfg.pnorm
    m, n = cfg.m, cfg.n
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
    # v_LB = v_LB.at[0].set(c_z)
    # v_UB = v_UB.at[0].set(c_z)
    # x_LB = x_l
    # x_UB = x_u

    # init_C = 1e4

    # Btx_UB, Btx_LB = interval_bound_prop(Bt, x_l, x_u)
    # if jnp.any(Btx_UB < Btx_LB):
    #     raise AssertionError('Btx upper/lower bounds are invalid')

    # beta = jnp.ones(K_max + 1)
    # gamma = jnp.zeros(K_max + 1)
    # for k in range(1, K_max+1):
    #     beta = beta.at[k].set(.5 * (1 + jnp.sqrt(1 + 4 * jnp.power(beta[k-1], 2))))
    #     gamma = gamma.at[k].set((beta[k-1] - 1) / beta[k])

    # log.info(f'beta: {beta}')
    # log.info(f'gamma: {gamma}')

    z = {}
    v = {}
    # z, y, v = {}, {}, {}

    # w1, w2 = {}, {}

    # model, x = Init_model()

    # Deltas = []
    # Delta_bounds = []
    # Delta_gaps = []
    # solvetimes = []
    # theory_tighter_fracs = []
    # x_out = jnp.zeros((K_max, m))

    # obj_scaling = cfg.obj_scaling.default

    # for k in range(1, K_max+1):
    #     log.info(f'----K={k}----')
    #     y_LB, y_UB, z_LB, z_UB, v_LB, v_UB = BoundPreprocessing(k, At, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, Btx_LB, Btx_UB, gamma, lambda_t)

    #     # if cfg.theory_bounds:
    #     #     z_LB, z_UB, theory_tight_frac = theory_bounds(k, A, t, lambd, c_z, z_LB, z_UB, x_LB, x_UB, init_C)
    #     #     theory_tighter_fracs.append(theory_tight_frac)

    #     if cfg.opt_based_tightening:
    #         y_LB, y_UB, z_LB, z_UB = BoundTightY(k, At, Bt, gamma, lambda_t, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, x_l, x_u)
    #         if jnp.any(y_LB > y_UB):
    #             raise AssertionError('y bounds invalid after bound tight y')
    #         if jnp.any(z_LB > z_UB):
    #             raise AssertionError('z bounds invalid after bound tight y + softthresholded')

    #         v_LB, v_UB = BoundTightV(k, At, Bt, gamma, lambda_t, y_LB, y_UB, z_LB, z_UB, v_LB, v_UB, x_l, x_u)
    #         if jnp.any(v_LB > v_UB):
    #             raise AssertionError('v bounds invalid after bound tight v')

    #     log.info(jnp.max(jnp.abs(z_UB[k] - z_UB[k-1])))


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
    m, n = cfg.m, cfg.n
    log.info(cfg)

    A = generate_data(cfg)
    A_eigs = jnp.real(jnp.linalg.eigvals(A.T @ A))
    log.info(f'eigenvalues of ATA: {A_eigs}')

    t = cfg.t

    log.info(f't={t}')

    if cfg.x.type == 'box':
        x_l = cfg.x.l * jnp.ones(m)
        x_u = cfg.x.u * jnp.ones(m)

    if cfg.lambd.val == 'adaptive':
        center = x_u - x_l
        lambd = cfg.lambd.scalar * jnp.max(jnp.abs(A.T @ center))
    else:
        lambd = cfg.lambd.val
    log.info(f'lambda: {lambd}')
    lambda_t = lambd * t
    log.info(f'lambda * t: {lambda_t}')

    if cfg.z0.type == 'lstsq':
        c_z = lstsq_sol(cfg, A, lambd, x_l, x_u)
    elif cfg.z0.type == 'zero':
        c_z = jnp.zeros(n)

    FISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u)


def run(cfg):
    if cfg.problem_type == 'random':
        random_FISTA_run(cfg)
    elif cfg.problem_type == 'sparse_coding':
        sparse_coding_FISTA_run(cfg)

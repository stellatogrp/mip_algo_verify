import copy
import logging

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
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
    "font.size": 20,
    "figure.figsize": (9, 6)})


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3
    absA = jnp.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_upper, Ax_lower


def BoundPreprocessing(k, At, y_LB, y_UB, z_LB, z_UB, Btx_LB, Btx_UB):
    Atzk_UB, Atzk_LB = interval_bound_prop(At, z_LB[k-1], z_UB[k-1])
    yk_UB = Atzk_UB + Btx_UB
    yk_LB = Atzk_LB + Btx_LB

    if jnp.any(yk_UB < yk_LB):
        raise AssertionError('basic y bound prop failed')

    return yk_LB, yk_UB


def BuildRelaxedModel(K, At, Bt, lambda_t, c_z, x_l, x_u, y_LB, y_UB, z_LB, z_UB):
    n, m = Bt.shape

    At = np.asarray(At)
    Bt = np.asarray(Bt)

    model = gp.Model()
    model.Params.OutputFlag = 0

    x = model.addMVar(m, lb=x_l, ub=x_u)

    # NOTE: we do NOT have bounds on zk yet, so only bound up to zk-1 and prop forward to yk
    z = model.addMVar((K, n), lb=z_LB[:K], ub=z_UB[:K])
    y = model.addMVar((K+1, n), lb=y_LB[:K+1], ub=y_UB[:K+1])

    for k in range(1, K+1):
        model.addConstr(y[k] == At @ z[k-1] + Bt @ x)

    # NOTE: stop at k-1 for the soft-thresholding relaxation and use the affine only to connect to yk
    for k in range(1, K):
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
    return model, y


def compute_lI(w, x, lambda_t, Lhat, Uhat, I, Icomp):
    if I.shape[0] == 0:
        return jnp.sum(jnp.multiply(w, Uhat)) - lambda_t
    if Icomp.shape[0] == 0:
        return jnp.sum(jnp.multiply(w, Lhat)) - lambda_t

    w_I = w[I]
    w_Icomp = w[Icomp]

    Lhat_I = Lhat[I]
    Uhat_I = Uhat[Icomp]

    return jnp.sum(jnp.multiply(w_I, Lhat_I)) + jnp.sum(jnp.multiply(w_Icomp, Uhat_I)) - lambda_t


def compute_v_pos(wi, xi, lambda_t, Lhat, Uhat):
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

    lI = compute_lI(wi, xi, lambda_t, Lhat, Uhat, I, jnp.array(list(Icomp)))
    # log.info(f'original lI: {lI}')
    if lI < 0:
        return None, None, None, None

    for h in filtered_idx:
        Itest = jnp.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        # log.info(Itest)
        # log.info(Icomp_test)

        lI_new = compute_lI(wi, xi, lambda_t, Lhat, Uhat, Itest.astype(jnp.integer), jnp.array(list(Icomp_test)))
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


def add_pos_conv_cuts(cfg, k, i, At, Bt, lambda_t, z_LB, z_UB, x_LB, x_UB, z, x, z_out):
    n, m = Bt.shape
    L_hat = jnp.zeros((m + n))
    U_hat = jnp.zeros((m + n))

    zkminus1_LB = z_LB[k-1]
    zkminus1_UB = z_UB[k-1]

    Ati = At[i]
    Bti = Bt[i]

    xi = jnp.hstack([z, x])
    wi = jnp.hstack([Ati, Bti])

    for j in range(n):
        if Ati[j] >= 0:
            L_hat = L_hat.at[j].set(zkminus1_LB[j])
            U_hat = U_hat.at[j].set(zkminus1_UB[j])
        else:
            L_hat = L_hat.at[j].set(zkminus1_UB[j])
            U_hat = U_hat.at[j].set(zkminus1_LB[j])

    for j in range(m):
        if Bti[j] >= 0:
            L_hat = L_hat.at[n + j].set(x_LB[j])
            U_hat = U_hat.at[n + j].set(x_UB[j])
        else:
            L_hat = L_hat.at[n + j].set(x_UB[j])
            U_hat = U_hat.at[n + j].set(x_LB[j])

    # log.info(L_hat)

    Iint, rhs, lI, h = compute_v_pos(wi, xi, lambda_t, L_hat, U_hat)

    if Iint is None:
        return None, None, None, None, None

    lhs = z_out[i]
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


def create_new_pos_constr(cfg, k, i, At, Bt, lambda_t, Iint, lI, h, z, x, Lhat, Uhat):
    Ati = At[i]
    Bti = Bt[i]

    zx_stack = gp.hstack([z[k-1], x])
    w = jnp.hstack([Ati, Bti])

    new_constr = 0
    for idx in Iint:
        new_constr += w[idx] * (zx_stack[idx] - Lhat[idx])
    new_constr += (lI - lambda_t) / (Uhat[h] - Lhat[h]) * (zx_stack[h] - Lhat[h])

    return z[k][i] <= new_constr


def compute_uI(w, x, lambda_t, Lhat, Uhat, I, Icomp):
    if I.shape[0] == 0:
        return jnp.sum(jnp.multiply(w, Lhat)) + lambda_t
    if Icomp.shape[0] == 0:
        return jnp.sum(jnp.multiply(w, Uhat)) + lambda_t

    w_I = w[I]
    w_Icomp = w[Icomp]

    Lhat_I = Lhat[Icomp]
    Uhat_I = Uhat[I]

    return jnp.sum(jnp.multiply(w_I, Uhat_I)) + jnp.sum(jnp.multiply(w_Icomp, Lhat_I)) + lambda_t


def compute_v_neg(wi, xi, lambda_t, Lhat, Uhat):
    idx = jnp.arange(wi.shape[0])
    # log.info(idx)

    filtered_idx = jnp.array([j for j in idx if wi[j] != 0 and jnp.abs(Lhat[j] - Uhat[j]) > 1e-7])
    # log.info(filtered_idx)

    def key_func(j):
        return (xi[j] - Uhat[j]) / (Lhat[j] - Uhat[j])

    keys = jnp.array([key_func(j) for j in filtered_idx])
    # log.info(keys)
    sorted_idx = jnp.argsort(keys)  # should this be in reverse order?
    # sorted_idx = jnp.argsort(keys)[::-1]
    filtered_idx = filtered_idx[sorted_idx]

    # log.info(filtered_idx)

    I = jnp.array([])
    Icomp = set(range(wi.shape[0]))

    # log.info(Icomp)

    uI = compute_uI(wi, xi, lambda_t, Lhat, Uhat, I, jnp.array(list(Icomp)))
    # log.info(f'original lI: {lI}')
    if uI > 0:
        return None, None, None, None

    for h in filtered_idx:
        Itest = jnp.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        # log.info(Itest)
        # log.info(Icomp_test)

        uI_new = compute_uI(wi, xi, lambda_t, Lhat, Uhat, Itest.astype(jnp.integer), jnp.array(list(Icomp_test)))
        # log.info(lI_new)
        if uI_new > 0:
            Iint = I.astype(jnp.integer)
            # log.info(f'h={h}')
            # log.info(f'uI before and after: {uI}, {uI_new}')
            rhs = jnp.sum(jnp.multiply(wi[Iint], xi[Iint])) + uI / (Lhat[int(h)] - Uhat[int(h)]) * (xi[int(h)] - Uhat[int(h)])
            return Iint, rhs, uI, int(h)

        I = Itest
        Icomp = Icomp_test
        uI = uI_new
    else:
        return None, None, None, None


def add_neg_conv_cuts(cfg, k, i, At, Bt, lambda_t, z_LB, z_UB, x_LB, x_UB, z, x, z_out):
    n, m = Bt.shape
    L_hat = jnp.zeros((m + n))
    U_hat = jnp.zeros((m + n))

    zkminus1_LB = z_LB[k-1]
    zkminus1_UB = z_UB[k-1]

    Ati = At[i]
    Bti = Bt[i]

    xi = jnp.hstack([z, x])
    wi = jnp.hstack([Ati, Bti])

    for j in range(n):
        if Ati[j] >= 0:
            L_hat = L_hat.at[j].set(zkminus1_LB[j])
            U_hat = U_hat.at[j].set(zkminus1_UB[j])
        else:
            L_hat = L_hat.at[j].set(zkminus1_UB[j])
            U_hat = U_hat.at[j].set(zkminus1_LB[j])

    for j in range(m):
        if Bti[j] >= 0:
            L_hat = L_hat.at[n + j].set(x_LB[j])
            U_hat = U_hat.at[n + j].set(x_UB[j])
        else:
            L_hat = L_hat.at[n + j].set(x_UB[j])
            U_hat = U_hat.at[n + j].set(x_LB[j])

    Iint, rhs, uI, h = compute_v_neg(wi, xi, lambda_t, L_hat, U_hat)

    if Iint is None:
    # if Iint.shape[0] == 0:
        return None, None, None, None, None

    lhs = z_out[i]
    if lhs < rhs - 1e-6:
        log.info('found a violated cut')
        log.info(f'with uI = {uI}')
        log.info(f'and I = {Iint}')
        # exit(0)

    if lhs < rhs - 1e-6:
        return Iint, uI, h, L_hat, U_hat
    else:
        return None, None, None, None, None


def create_new_neg_constr(cfg, k, i, At, Bt, lambda_t, Iint, uI, h, z, x, Lhat, Uhat):
    Ati = At[i]
    Bti = Bt[i]

    zx_stack = gp.hstack([z[k-1], x])
    w = jnp.hstack([Ati, Bti])

    new_constr = 0
    for idx in Iint:
        new_constr += w[idx] * (zx_stack[idx] - Uhat[idx])
    new_constr += (uI + lambda_t) / (Lhat[h] - Uhat[h]) * (zx_stack[h] - Uhat[h])

    return z[k][i] >= new_constr


def BoundTightY(k, At, Bt, lambda_t, c_z, x_l, x_u, y_LB, y_UB, z_LB, z_UB):
    log.info('bound tightening for y')
    model, y = BuildRelaxedModel(k, At, Bt, lambda_t, c_z, x_l, x_u, y_LB, y_UB, z_LB, z_UB)
    n = At.shape[0]
    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        for i in range(n):
            model.setObjective(y[k, i], sense)
            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print('bound tighting failed, GRB model status:', model.status)
                exit(0)
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


def sample_radius(cfg, A, t, lambd, c_z, x_LB, x_UB, C_norm=2):
    sample_idx = jnp.arange(cfg.samples.init_dist_N)
    m, n = A.shape

    def z_sample(i):
        return c_z

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(m,), minval=x_LB, maxval=x_UB)

    # z_samples = jax.vmap(z_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    distances = jnp.zeros(cfg.samples.init_dist_N)
    for i in trange(cfg.samples.init_dist_N):
        z = cp.Variable(n)
        x_samp = x_samples[i]
        obj = cp.Minimize(.5 * cp.sum_squares(A @ z - x_samp) + lambd * cp.norm(z, 1))
        prob = cp.Problem(obj)
        prob.solve()

        distances = distances.at[i].set(np.linalg.norm(z.value - c_z, ord=C_norm))

    # log.info(distances)

    return jnp.max(distances), z.value, x_samp


def init_dist(cfg, A, t, lambd, c_z, x_LB, x_UB, C_norm=2):
    SM_initC, z_samp, x_samp = sample_radius(cfg, A, t, lambd, c_z, x_LB, x_UB, C_norm=C_norm)
    log.info(SM_initC)

    m, n = A.shape
    model = gp.Model()

    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    At = np.asarray(At)
    Bt = np.asarray(Bt)

    bound_M = cfg.star_bound_M
    z_star = model.addMVar(n, lb=-bound_M, ub=bound_M)
    x = model.addMVar(m, lb=x_LB, ub=x_UB)
    z0 = model.addMVar(n, lb=c_z, ub=c_z)
    w1 = model.addMVar(n, vtype=gp.GRB.BINARY)
    w2 = model.addMVar(n, vtype=gp.GRB.BINARY)

    M = cfg.init_dist_M
    z_star.Start = z_samp
    x.Start = x_samp

    model.setParam('TimeLimit', cfg.timelimit)
    model.setParam('MIPGap', cfg.mipgap)

    y_star = At @ z_star + Bt @ x

    for i in range(n):
        model.addConstr(z_star[i] >= y_star[i] - lambda_t)
        model.addConstr(z_star[i] <= y_star[i] + lambda_t)

        model.addConstr(z_star[i] <= y_star[i] - lambda_t + (2 * lambda_t)*(1-w1[i]))
        model.addConstr(z_star[i] >= y_star[i] + lambda_t + (-2 * lambda_t) * (1-w2[i]))

        model.addConstr(z_star[i] <= M * w1[i])
        model.addConstr(z_star[i] >= -M * w2[i])

        model.addConstr(w1[i] + w2[i] <= 1)

    if C_norm == 2:
        obj = (z_star - z0) @ (z_star - z0)
        model.setObjective(obj, gp.GRB.MAXIMIZE)
        model.optimize()

        # max_rad = np.sqrt(model.objVal)
        incumbent = np.sqrt(model.objVal)
        max_rad = np.sqrt(model.objBound)

    elif C_norm == 1:
        y = (z_star - z0)
        up = model.addMVar(n, lb=0, ub=bound_M)
        un = model.addMVar(n, lb=0, ub=bound_M)
        omega = model.addMVar(n, vtype=gp.GRB.BINARY)

        model.addConstr(up - un == y)
        for i in range(n):
            model.addConstr(up[i] <= bound_M * omega[i])
            model.addConstr(un[i] <= bound_M * (1-omega[i]))

        model.setObjective(gp.quicksum(up + un), gp.GRB.MAXIMIZE)
        model.optimize()

        # max_rad = model.objVal
        incumbent = model.objVal
        max_rad = model.objBound

    log.info(f'sample max init C: {SM_initC}')
    log.info(f'run time: {model.Runtime}')
    log.info(f'incumbent sol: {incumbent}')
    log.info(f'miqp max radius bound: {max_rad}')

    return max_rad


def theory_bounds(k, A, t, lambd, c_z, z_LB, z_UB, x_LB, x_UB, init_C):
    log.info(f'-theory bound for k={k}-')
    if k == 1:
        return z_LB, z_UB, 0

    ATA = A.T @ A
    n = ATA.shape[0]
    # mu = jnp.min(jnp.real(jnp.linalg.eigvals(ATA)))
    # log.info(f'mu = {mu}')

    # frac = (1- mu / t) / (1 + mu / t)

    # const = 2 / t * jnp.sqrt(jnp.abs(jnp.power(frac, k))) * init_C
    # log.info(f'theory bound on fp resid: {const}')

    log.info('remember this bound only works when t=1/L')
    const = 2 * init_C / np.sqrt((k-1) * (k+2))

    theory_tight_count = 0
    for i in range(n):
        if z_LB[k-1, i] - const > z_LB[k, i]:
            theory_tight_count += 1
            log.info(f'min, before: {z_LB[k, i]}, after {z_LB[k-1, i] - const}')
        z_LB = z_LB.at[k, i].set(max(z_LB[k, i], z_LB[k-1, i] - const))

        if z_UB[k-1, i] + const < z_UB[k, i]:
            theory_tight_count += 1
            log.info(f'max, before: {z_UB[k, i]}, after {z_UB[k-1, i] + const}')
        z_UB = z_UB.at[k, i].set(min(z_UB[k, i], z_UB[k-1, i] + const))


    return z_LB, z_UB, theory_tight_count / (2 * n)


def ISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u):

    def Init_model():
        model = gp.Model()
        model.setParam('TimeLimit', cfg.timelimit)
        model.setParam('MIPGap', cfg.mipgap)
        model.setParam('MIPFocus', cfg.mipfocus)
        # model.setParam('OBBT', 0)
        # model.setParam('Cuts', 0)

        x = model.addMVar(m, lb=x_l, ub=x_u)
        z[0] = model.addMVar(n, lb=c_z, ub=c_z)  # if non singleton, change here

        model.update()
        return model, x

    def ModelNextStep(model, k, At, Bt, lambda_t, c_z, y_LB, y_UB, z_LB, z_UB, obj_scaling=cfg.obj_scaling.default):
        obj_constraints = []

        y[k] = model.addMVar(n, lb=y_LB[k], ub=y_UB[k])
        z[k] = model.addMVar(n, lb=z_LB[k], ub=z_UB[k])

        # y[k] = model.addMVar(n, lb=-np.inf, ub=np.inf)
        # z[k] = model.addMVar(n, lb=-np.inf, ub=np.inf)

        # affine constraints
        model.addConstr(y[k] == At @ z[k-1] + Bt @ x)

        # soft-thresholding
        for i in range(n):
            if y_LB[k, i] >= lambda_t:
                model.addConstr(z[k][i] == y[k][i] - lambda_t)

            elif y_UB[k, i] <= -lambda_t:
                model.addConstr(z[k][i] == y[k][i] + lambda_t)

            elif y_LB[k, i] >= -lambda_t and y_UB[k, i] <= lambda_t:
                model.addConstr(z[k][i] == 0.0)

            else:
                if y_LB[k, i] < -lambda_t and y_UB[k, i] > lambda_t:
                    w1[k, i] = model.addVar(vtype=GRB.BINARY)
                    w2[k, i] = model.addVar(vtype=GRB.BINARY)
                    # add back
                    model.addConstr(z[k][i] >= y[k][i] - lambda_t)
                    model.addConstr(z[k][i] <= y[k][i] + lambda_t)

                    model.addConstr(z[k][i] <= z_UB[k, i]/(y_UB[k, i] + lambda_t) * (y[k][i] + lambda_t))
                    model.addConstr(z[k][i] >= z_LB[k, i]/(y_LB[k, i] - lambda_t) * (y[k][i] - lambda_t))

                    # Upper right part: w1 = 1, y >= lambda_t
                    # model.addConstr(z[k][i] <= y[k][i] - lambda_t + (lambda_t + z_UB[k, i] - y_LB[k, i])*(1-w1[k, i]))  # check this
                    model.addConstr(y[k][i] >= lambda_t + (y_LB[k, i] - lambda_t)*(1-w1[k, i]))
                    model.addConstr(y[k][i] <= lambda_t + (y_UB[k, i] - lambda_t)*w1[k, i])

                    # Lower left part: w2 = 1, y <= -lambda_t
                    # model.addConstr(z[k][i] >= y[k][i] + lambda_t + (z_LB[k, i] + y_UB[k, i])*(1-w2[k, i]))  # check this
                    model.addConstr(y[k][i] <= -lambda_t + (y_UB[k, i] + lambda_t)*(1-w2[k, i]))
                    model.addConstr(y[k][i] >= -lambda_t + (y_LB[k, i] + lambda_t)*w2[k, i])

                    # model.addConstr(z[k][i] <= y[k][i] - lambda_t + (z_UB[k, i] - z_LB[k, i])*(1-w1[k, i]))  # can we just use z_UB?
                    # model.addConstr(z[k][i] >= y[k][i] + lambda_t + (z_LB[k, i] - z_UB[k, i])*(1-w2[k, i]))  # can we just use z_LB?

                    # model.addConstr(z[k][i] <= y[k][i] - lambda_t + (z_UB[k, i])*(1-w1[k, i]))  # can use 2 lambda_t and -2lambda_t
                    # model.addConstr(z[k][i] >= y[k][i] + lambda_t + (z_LB[k, i])*(1-w2[k, i]))

                    model.addConstr(z[k][i] <= y[k][i] - lambda_t + (2 * lambda_t)*(1-w1[k, i]))  # can use 2 lambda_t and -2lambda_t
                    model.addConstr(z[k][i] >= y[k][i] + lambda_t + (-2 * lambda_t)*(1-w2[k, i]))

                    # If both binary vars are 0, then this forces z = 0
                    # model.addConstr(z[k][i] <= (z_UB[k, i])*(w1[k, i] + w2[k, i]))
                    # model.addConstr(z[k][i] >= (z_LB[k, i])*(w1[k, i] + w2[k, i]))
                    model.addConstr(z[k][i] <= z_UB[k, i] * w1[k, i])
                    model.addConstr(z[k][i] >= z_LB[k, i] * w2[k, i])

                    # The left and right part cannot be hold at the same time (improve LP relaxation)
                    model.addConstr(w1[k, i] + w2[k, i] <= 1)

                elif -lambda_t <= y_LB[k, i] <= lambda_t and y_UB[k, i] > lambda_t:
                    w1[k, i] = model.addVar(vtype=GRB.BINARY)
                    model.update()

                    # add back
                    model.addConstr(z[k][i] >= 0)
                    model.addConstr(z[k][i] <= z_UB[k, i]/(y_UB[k, i] - y_LB[k, i])*(y[k][i] - y_LB[k, i]))
                    model.addConstr(z[k][i] >= y[k][i] - lambda_t)

                    # Upper right part: w1 = 1, y >= lambda_t
                    # model.addConstr(z[k][i] <= y[k][i] - lambda_t + (lambda_t + z_UB[k, i] - y_LB[k, i])*(1-w1[k, i]))
                    model.addConstr(y[k][i] >= lambda_t + (y_LB[k, i] - lambda_t)*(1-w1[k, i]))
                    model.addConstr(y[k][i] <= lambda_t + (y_UB[k, i] - lambda_t)*w1[k, i])

                    # model.addConstr(z[k][i] <= y[k][i] - lambda_t + (z_UB[k, i])*(1-w1[k, i]))
                    model.addConstr(z[k][i] <= y[k][i] - lambda_t + (2 * lambda_t)*(1-w1[k, i]))
                    model.addConstr(z[k][i] <= z_UB[k, i] * w1[k, i])

                elif -lambda_t <= y_UB[k, i] <= lambda_t and y_LB[k, i] < -lambda_t:
                    w2[k, i] = model.addVar(vtype=GRB.BINARY)
                    model.update()

                    # add back
                    model.addConstr(z[k][i] <= 0)
                    model.addConstr(z[k][i] >= z_LB[k, i]/(y_LB[k, i] - y_UB[k, i])*(y[k][i]- y_UB[k, i]))
                    model.addConstr(z[k][i] <= y[k][i] + lambda_t)

                    # Lower left part: w2 = 1, y <= -lambda_t
                    # model.addConstr(z[k][i] >= y[k][i] + lambda_t + (z_LB[k, i] + y_UB[k, i])*(1-w2[k, i]))
                    model.addConstr(y[k][i] <= -lambda_t + (y_UB[k, i] + lambda_t)*(1-w2[k, i]))
                    model.addConstr(y[k][i] >= -lambda_t + (y_LB[k, i] + lambda_t)*w2[k, i])

                    # model.addConstr(z[k][i] >= y[k][i] + lambda_t + (z_LB[k, i])*(1-w2[k, i]))
                    model.addConstr(z[k][i] >= y[k][i] + lambda_t + (-2 * lambda_t)*(1-w2[k, i]))
                    model.addConstr(z[k][i] >= z_LB[k, i] * w2[k, i])
                else:
                    raise RuntimeError('Unreachable code', y_LB[k, i], y_UB[k, i], lambda_t)

        # setting up for objective
        U = z_UB[k] - z_LB[k-1]
        L = z_LB[k] - z_UB[k-1]

        v = model.addMVar(n, vtype=gp.GRB.BINARY)
        up = model.addMVar(n, ub=jnp.abs(U))
        un = model.addMVar(n, ub=jnp.abs(L))

        # up = model.addMVar(n, ub=np.inf)
        # un = model.addMVar(n, ub=np.inf)
        # v = {}
        # up = {}
        # un = {}

        if pnorm == 1 or pnorm == 'inf':
            obj_constraints.append(model.addConstr(up - un == z[k] - z[k-1]))

            # for i in range(n):
            #     up[i] = model.addVar(ub=jnp.abs(U)[i])
            #     un[i] = model.addVar(ub=jnp.abs(L)[i])
            #     v[i] = model.addVar(vtype=gp.GRB.BINARY)

            #     obj_constraints.append(up[i] - un[i] == z[k][i] - z[k-1][i])

            for i in range(n):
                obj_constraints.append(up[i] <= np.abs(z_UB[k, i] - z_LB[k-1, i]) * v[i])
                obj_constraints.append(un[i] <= np.abs(z_LB[k, i] - z_UB[k-1, i]) * (1 - v[i]))

            for i in range(n):
                if L[i] >= 0:
                    obj_constraints.append(model.addConstr(up[i] == z[k][i] - z[k-1][i]))
                    obj_constraints.append(model.addConstr(un[i] == 0))
                elif U[i] < 0:
                    obj_constraints.append(model.addConstr(un[i] == z[k-1][i] - z[k][i]))
                    obj_constraints.append(model.addConstr(up[i] == 0))
                else:
                    obj_constraints.append(model.addConstr(up[i] - un[i] == z[k][i] - z[k-1][i]))
                    obj_constraints.append(model.addConstr(up[i] <= jnp.abs(U[i]) * v[i]))
                    obj_constraints.append(model.addConstr(un[i] <= jnp.abs(L[i]) * (1-v[i])))

        if pnorm == 1:
            model.setObjective(1 / obj_scaling * gp.quicksum(up + un), GRB.MAXIMIZE)
        elif pnorm == 'inf':
            M = jnp.maximum(jnp.max(jnp.abs(U)), jnp.max(jnp.abs(L)))
            q = model.addVar(ub=M)
            gamma = model.addMVar(n, vtype=gp.GRB.BINARY)

            for i in range(n):
                obj_constraints.append(model.addConstr(q >= up[i] + un[i]))
                obj_constraints.append(model.addConstr(q <= up[i] + un[i] + M * (1 - gamma[i])))

            obj_constraints.append(model.addConstr(gp.quicksum(gamma) == 1))
            model.setObjective(1 / obj_scaling * q, gp.GRB.MAXIMIZE)

        model.update()
        if cfg.exact_conv_relax.use_in_l1_rel:
            rel_model = model.relax()
            rel_model.optimize()

            rel_z = np.array([])
            rel_x = np.array([])

            rel_z_out = np.array([])

            for var in z[k-1]:
                rel_z = np.append(rel_z, rel_model.getVarByName(var.VarName.item()).X)

            for var in z[k]:
                rel_z_out = np.append(rel_z_out, rel_model.getVarByName(var.VarName.item()).X)

            for var in x:
                rel_x = np.append(rel_x, rel_model.getVarByName(var.VarName.item()).X)


            for i in range(n):
                log.info('--computing conv cuts--')

                if (k, i) in w1 and rel_z_out[i] > 0:
                    Iint, lI, h, L_hat, U_hat = add_pos_conv_cuts(cfg, k, i, At, Bt, lambda_t, z_LB, z_UB, x_LB, x_UB, rel_z, rel_x, rel_z_out)

                    if Iint is not None:
                        log.info(f'new lI constraint added with {(k, i)}')
                        model.addConstr(create_new_pos_constr(cfg, k, i, At, Bt, lambda_t, Iint, lI, h, z, x, L_hat, U_hat))

                if (k, i) in w2 and rel_z_out[i] < 0:
                    Iint, uI, h, L_hat, U_hat = add_neg_conv_cuts(cfg, k, i, At, Bt, lambda_t, z_LB, z_UB, x_LB, x_UB, rel_z, rel_x, rel_z_out)

                    if Iint is not None:
                        log.info(f'new uI constraint added with {(k, i)}')
                        model.addConstr(create_new_neg_constr(cfg, k, i, At, Bt, lambda_t, Iint, uI, h, z, x, L_hat, U_hat))
        model.update()
        model.optimize()

        for constr in obj_constraints:
            try:
                model.remove(constr)
            except gp.GurobiError:
                pass

        # return model.objVal * obj_scaling, model.Runtime, x.X
        return model.objVal * obj_scaling, model.objBound * obj_scaling, model.MIPGap, model.Runtime, x.X

    max_sample_resids = samples(cfg, A, lambd, t, c_z, x_l, x_u)
    log.info(f'max sample resids: {max_sample_resids}')

    # max_sample_resids = samples_diffK(cfg, A, lambd, t, c_z, x_l, x_u)
    # log.info(f'max sample resids with diff samples per K: {max_sample_resids}')

    pnorm = cfg.pnorm
    m, n = cfg.m, cfg.n
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T

    At = np.asarray(At)
    Bt = np.asarray(Bt)
    lambda_t = lambd * t

    K_max = cfg.K_max

    z_LB = jnp.zeros((K_max + 1, n))
    z_UB = jnp.zeros((K_max + 1, n))
    y_LB = jnp.zeros((K_max + 1, n))
    y_UB = jnp.zeros((K_max + 1, n))

    z_LB = z_LB.at[0].set(c_z)
    z_UB = z_UB.at[0].set(c_z)
    x_LB = x_l
    x_UB = x_u

    init_C = init_dist(cfg, A, t, lambd, c_z, x_LB, x_UB, C_norm=cfg.C_norm)

    Btx_UB, Btx_LB = interval_bound_prop(Bt, x_l, x_u)
    if jnp.any(Btx_UB < Btx_LB):
        raise AssertionError('Btx upper/lower bounds are invalid')

    log.info(Btx_LB)
    log.info(Btx_UB)

    z, y = {}, {}

    # up, un, v = {}, {}, {}
    w1, w2 = {}, {}

    # gamma, q = {}, {}

    # obj_constraints = []

    model, x = Init_model()

    Deltas = []
    Delta_bounds = []
    Delta_gaps = []
    solvetimes = []
    theory_tighter_fracs = []
    x_out = jnp.zeros((K_max, m))

    obj_scaling = cfg.obj_scaling.default

    for k in range(1, K_max+1):
        log.info(f'----K={k}----')
        yk_LB, yk_UB = BoundPreprocessing(k, At, y_LB, y_UB, z_LB, z_UB, Btx_LB, Btx_UB)
        y_LB = y_LB.at[k].set(yk_LB)
        y_UB = y_UB.at[k].set(yk_UB)
        z_LB = z_LB.at[k].set(soft_threshold(yk_LB, lambda_t))
        z_UB = z_UB.at[k].set(soft_threshold(yk_UB, lambda_t))

        if cfg.theory_bounds:
            z_LB, z_UB, theory_tight_frac = theory_bounds(k, A, t, lambd, c_z, z_LB, z_UB, x_LB, x_UB, init_C)
            theory_tighter_fracs.append(theory_tight_frac)

        if cfg.opt_based_tightening:
            for _ in range(cfg.num_obbt_iter):
                y_LB, y_UB, z_LB, z_UB = BoundTightY(k, At, Bt, lambda_t, c_z, x_l, x_u, y_LB, y_UB, z_LB, z_UB)
                if jnp.any(y_LB > y_UB):
                    raise AssertionError('y bounds invalid after bound tight y')
                if jnp.any(z_LB > z_UB):
                    raise AssertionError('z bounds invalid after bound tight y + softthresholded')

        result, bound, opt_gap, time, xval= ModelNextStep(model, k, At, Bt, lambda_t, c_z, y_LB, y_UB, z_LB, z_UB, obj_scaling=obj_scaling)
        x_out = x_out.at[k-1].set(xval)
        log.info(result)
        log.info(xval)

        Deltas.append(result)
        Delta_bounds.append(bound)
        Delta_gaps.append(opt_gap)
        solvetimes.append(time)

        if cfg.obj_scaling.val == 'adaptive':
            obj_scaling = result

        if cfg.postprocessing:
            # Dk = jnp.sum(jnp.array(Deltas))
            Dk = jnp.sum(jnp.array(Delta_bounds))
            for i in range(n):
                z_LB = z_LB.at[k, i].set(max(c_z[i] - Dk, soft_threshold(y_LB[k, i], lambda_t)))
                z_UB = z_UB.at[k, i].set(min(c_z[i] + Dk, soft_threshold(y_UB[k, i], lambda_t)))
                z[k][i].LB = z_LB[k, i]
                z[k][i].UB = z_UB[k, i]

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
        ax.set_title(rf'ISTA VP, $m={cfg.m}$, $n={cfg.n}$')

        ax.legend()
        plt.tight_layout()

        plt.savefig('resids.pdf')

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
        ax.set_title(rf'ISTA VP, $m={cfg.m}$, $n={cfg.n}$')

        ax.legend()

        plt.savefig('times.pdf')
        plt.clf()
        plt.cla()
        plt.close()

        x_out_plot = x_out.T

        plt.imshow(x_out_plot, cmap='viridis')
        plt.colorbar()

        plt.xlabel(r'$K$')
        plt.savefig('x_heatmap.pdf')

        df = pd.DataFrame(x_out_plot)
        df.to_csv('x_heatmap.csv', index=False, header=False)

        plt.clf()
        plt.cla()
        plt.close()

        log.info(f'max_sample_resids: {max_sample_resids}')
        log.info(f'Deltas: {Deltas}')
        log.info(f'times: {solvetimes}')
        log.info(f'theory tighter fracs: {theory_tighter_fracs}')

    diffs = jnp.array(Deltas) - jnp.array(max_sample_resids)
    log.info(f'deltas - max_sample_resids: {diffs}')
    if jnp.any(diffs < 0):
        log.info('error, SM > VP')


def soft_threshold(x, gamma):
    return jnp.sign(x) * jax.nn.relu(jnp.abs(x) - gamma)


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


def samples(cfg, A, lambd, t, c_z, x_l, x_u):
    n = cfg.n
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    sample_idx = jnp.arange(cfg.samples.N)

    def z_sample(i):
        return c_z

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.m,), minval=x_l, maxval=x_u)

    z_samples = jax.vmap(z_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    def ista_resids(i):
        return ISTA_alg(At, Bt, z_samples[i], x_samples[i], lambda_t, cfg.K_max, pnorm=cfg.pnorm)

    _, sample_resids = jax.vmap(ista_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def samples_diffK(cfg, A, lambd, t, c_z, x_l, x_u):
    K_max = cfg.K_max

    n = cfg.n
    # t = cfg.t
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = lambd * t

    def z_sample(i):
        return c_z

    sample_idx = jnp.arange(cfg.samples.N)
    z_samples = jax.vmap(z_sample)(sample_idx)

    maxes = []

    for k in range(1, K_max+1):
        log.info(f'computing samples for k={k}')
        def x_sample(i):
            key = jax.random.PRNGKey(cfg.samples.x_seed_offset * k + i)
            # TODO add the if, start with box case only
            return jax.random.uniform(key, shape=(cfg.m,), minval=x_l, maxval=x_u)

        x_samples_k = jax.vmap(x_sample)(sample_idx)
        def ista_resids(i):
            return ISTA_alg(At, Bt, z_samples[i], x_samples_k[i], lambda_t, k, pnorm=cfg.pnorm)

        _, sample_resids = jax.vmap(ista_resids)(sample_idx)
        log.info(sample_resids)
        max_sample_k = jnp.max(sample_resids[:, -1])
        log.info(f'max: {max_sample_k}')
        maxes.append(max_sample_k)

    return jnp.array(maxes)


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


def random_ISTA_run(cfg):
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

    ISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u)


def sparse_coding_A(cfg):
    m, n = cfg.m, cfg.n
    key = jax.random.PRNGKey(cfg.A_rng_seed)

    key, subkey = jax.random.split(key)
    A = 1 / m * jax.random.normal(subkey, shape=(m, n))

    # A_mask = jax.random.bernoulli(key, p=cfg.x_star.A_mask_prob, shape=(m-1, n)).astype(jnp.float64)

    # masked_A = jnp.multiply(A[1:], A_mask)

    # A = A.at[1:].set(masked_A)
    # return A / jnp.linalg.norm(A, axis=0)
    A_mask = jax.random.bernoulli(key, p=cfg.x_star.A_mask_prob, shape=(m, n)).astype(jnp.float64)
    masked_A = jnp.multiply(A, A_mask)
    # log.info(masked_A)

    for i in range(n):
        Ai = masked_A[:, i]
        if jnp.linalg.norm(Ai) > 0:
            masked_A = masked_A.at[:, i].set(Ai / jnp.linalg.norm(Ai))

    # log.info(jnp.linalg.norm(masked_A, axis=0))
    # log.info(jnp.count_nonzero(masked_A.T @ masked_A))
    # exit(0)

    return masked_A


def sparse_coding_b_set(cfg, A):
    m, n = A.shape

    key = jax.random.PRNGKey(cfg.x_star.rng_seed)

    key, subkey = jax.random.split(key)
    x_star_set = cfg.x_star.std * jax.random.normal(subkey, shape=(n, cfg.x_star.num))

    key, subkey = jax.random.split(key)
    x_star_mask = jax.random.bernoulli(subkey, p=cfg.x_star.nonzero_prob, shape=(n, cfg.x_star.num))

    x_star = jnp.multiply(x_star_set, x_star_mask)
    # log.info(x_star)

    epsilon = cfg.x_star.epsilon_std * jax.random.normal(key, shape=(m, cfg.x_star.num))

    b_set = A @ x_star + epsilon

    # log.info(A @ x_star)
    # log.info(b_set)

    return b_set


def sparse_coding_ISTA_run(cfg):
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

    ISTA_verifier(cfg, A, lambd, t, c_z, x_l, x_u)


def run(cfg):
    if cfg.problem_type == 'random':
        random_ISTA_run(cfg)
    elif cfg.problem_type == 'sparse_coding':
        sparse_coding_ISTA_run(cfg)

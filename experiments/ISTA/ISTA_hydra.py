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


def BoundTightY(k, At, Bt, lambda_t, c_z, x_l, x_u, y_LB, y_UB, z_LB, z_UB):
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


def ISTA_verifier(cfg, A, c_z):

    def Init_model():
        model = gp.Model()

        x = model.addMVar(m, lb=x_l, ub=x_u)
        z[0] = model.addMVar(n, lb=c_z, ub=c_z)  # if non singleton, change here

        model.update()
        return model, x

    def ModelNextStep(model, k, At, Bt, lambda_t, c_z, y_LB, y_UB, z_LB, z_UB):
        obj_constraints = []

        y[k] = model.addMVar(n, lb=y_LB[k], ub=y_UB[k])
        z[k] = model.addMVar(n, lb=z_LB[k], ub=z_UB[k])

        # log.info(y_LB)
        # log.info(y_UB)

        # log.info(z_LB)
        # log.info(z_UB)

        for constr in obj_constraints:
            model.remove(constr)
        model.update()

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

        if pnorm == 1 or pnorm == 'inf':
            obj_constraints.append(model.addConstr(up - un == z[k] - z[k-1]))
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
            model.setObjective(cfg.obj_scaling * gp.quicksum(up + un), GRB.MAXIMIZE)
        elif pnorm == 'inf':
            M = jnp.maximum(jnp.max(jnp.abs(U)), jnp.max(jnp.abs(L)))
            q = model.addVar(ub=M)
            gamma = model.addMVar(n, vtype=gp.GRB.BINARY)

            for i in range(n):
                obj_constraints.append(model.addConstr(q >= up[i] + un[i]))
                obj_constraints.append(model.addConstr(q <= up[i] + un[i] + M * (1 - gamma[i])))

            obj_constraints.append(model.addConstr(gp.quicksum(gamma) == 1))
            model.setObjective(cfg.obj_scaling * q, gp.GRB.MAXIMIZE)

        model.update()
        model.optimize()

        return model.objVal / cfg.obj_scaling, model.Runtime

    max_sample_resids = samples(cfg, A, c_z)

    pnorm = cfg.pnorm
    m, n = cfg.m, cfg.n
    t = cfg.t
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T

    At = np.asarray(At)
    Bt = np.asarray(Bt)
    lambda_t = cfg.lambd * cfg.t

    K_max = cfg.K_max

    if cfg.x.type == 'box':
        x_l = cfg.x.l * jnp.ones(m)
        x_u = cfg.x.u * jnp.ones(m)

    z_LB = jnp.zeros((K_max + 1, n))
    z_UB = jnp.zeros((K_max + 1, n))
    y_LB = jnp.zeros((K_max + 1, n))
    y_UB = jnp.zeros((K_max + 1, n))

    z_LB = z_LB.at[0].set(c_z)
    z_UB = z_UB.at[0].set(c_z)

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
    solvetimes = []
    for k in range(1, K_max+1):
        log.info(f'----K={k}----')
        yk_LB, yk_UB = BoundPreprocessing(k, At, y_LB, y_UB, z_LB, z_UB, Btx_LB, Btx_UB)
        y_LB = y_LB.at[k].set(yk_LB)
        y_UB = y_UB.at[k].set(yk_UB)
        z_LB = z_LB.at[k].set(soft_threshold(yk_LB, lambda_t))
        z_UB = z_UB.at[k].set(soft_threshold(yk_UB, lambda_t))

        # TODO: add theory bound in between
        y_LB, y_UB, z_LB, z_UB = BoundTightY(k, At, Bt, lambda_t, c_z, x_l, x_u, y_LB, y_UB, z_LB, z_UB)
        if jnp.any(y_LB > y_UB):
            raise AssertionError('y bounds invalid after bound tight y')
        if jnp.any(z_LB > z_UB):
            raise AssertionError('z bounds invalid after bound tight y + softthresholded')

        result, time = ModelNextStep(model, k, At, Bt, lambda_t, c_z, y_LB, y_UB, z_LB, z_UB)
        log.info(result)

        Deltas.append(result)
        solvetimes.append(time)

        Dk = jnp.sum(jnp.array(Deltas))
        for i in range(n):
            z_LB = z_LB.at[k, i].set(max(c_z[i] - Dk, soft_threshold(y_LB[k, i], lambda_t)))
            z_UB = z_UB.at[k, i].set(min(c_z[i] + Dk, soft_threshold(y_UB[k, i], lambda_t)))
            z[k][i].LB = z_LB[k, i]
            z[k][i].UB = z_UB[k, i]
        model.update()

        df = pd.DataFrame(Deltas)  # remove the first column of zeros
        df.to_csv('resids.csv', index=False, header=False)

        df = pd.DataFrame(solvetimes)
        df.to_csv('solvetimes.csv', index=False, header=False)

        # plotting resids so far
        fig, ax = plt.subplots()
        ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
        ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

        ax.set_xlabel(r'$K$')
        ax.set_ylabel('Fixed-point residual')
        ax.set_yscale('log')
        ax.set_title(rf'ISTA VP, $m={cfg.m}$, $n={cfg.n}$')

        ax.legend()

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


    log.info(f'max_sample_resids: {max_sample_resids}')
    log.info(f'Deltas: {Deltas}')
    log.info(f'times: {solvetimes}')

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


def samples(cfg, A, c_z):
    n = cfg.n
    t = cfg.t
    At = jnp.eye(n) - t * A.T @ A
    Bt = t * A.T
    lambda_t = cfg.lambd * cfg.t

    sample_idx = jnp.arange(cfg.samples.N)

    def z_sample(i):
        return c_z

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.m,), minval=cfg.x.l, maxval=cfg.x.u)

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


def lstsq_sol(cfg, A):
    m, n = cfg.m, cfg.n
    lambd = cfg.lambd

    if cfg.x.type == 'box':
        x_l = cfg.x.l
        x_u = cfg.x.u

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
    n = cfg.n
    log.info(cfg)

    A = generate_data(cfg)
    A_eigs = jnp.real(jnp.linalg.eigvals(A.T @ A))
    log.info(f'eigenvalues of ATA: {A_eigs}')

    z_lstsq = lstsq_sol(cfg, A)

    if cfg.z0.type == 'lstsq':
        c_z = z_lstsq
    elif cfg.z0.type == 'zero':
        c_z = jnp.zeros(n)

    lambda_t = cfg.lambd * cfg.t
    log.info(f'lambda * t: {lambda_t}')

    ISTA_verifier(cfg, A, c_z)


def run(cfg):
    random_ISTA_run(cfg)

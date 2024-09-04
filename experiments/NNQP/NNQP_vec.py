import logging
import time

import gurobipy as gp
import jax
import jax.numpy as jnp
import numpy as np

jnp.set_printoptions(precision=5)  # Print few decimal places
jnp.set_printoptions(suppress=True)  # Suppress scientific notation
jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


def VerifyPGD_withBounds_twostep(K, A, B, t, cfg, Deltas,
                                 y_LB, y_UB, z_LB, z_UB, x_LB, x_UB,
                                 zbar, ybar, xbar):
    n = cfg.n
    model = gp.Model()
    pnorm = cfg.pnorm

    var_shape = (K+1, n)
    # log.info(z_LB.shape)
    # log.info(z_UB.shape)
    # log.info(var_shape)
    z = model.addMVar(var_shape, lb=z_LB[:K+1], ub=z_UB[:K+1])
    y = model.addMVar(var_shape, lb=y_LB[:K+1], ub=y_UB[:K+1])
    x = model.addMVar(n, lb=x_LB, ub=x_UB)
    w = model.addMVar(var_shape, vtype=gp.GRB.BINARY)

    # affine step constraints
    for k in range(K):
        model.addConstr(y[k+1] == np.asarray(A) @ z[k] + np.asarray(B) @ x)

    # relu constraints
    for k in range(K):
        for i in range(n):
            if y_UB[k+1, i] < -0.00001:
                model.addConstr(z[k+1, i] == 0)
            elif y_LB[k+1, i] > 0.00001:
                model.addConstr(z[k+1, i] == y[k+1, i])
            else:
                model.addConstr(z[k+1, i] <= y_UB[k+1, i]/(y_UB[k+1, i] - y_LB[k+1, i]) * (y[k+1, i] - y_LB[k+1, i]))
                model.addConstr(z[k+1, i] >= y[k+1, i])
                model.addConstr(z[k+1, i] <= y[k+1, i] - y_LB[k+1, i] * (1 - w[k+1, i]))
                model.addConstr(z[k+1, i] <= y_UB[k+1, i] * w[k+1, i])

    if zbar is not None:
        z[:K-1].Start = zbar[:K-1]
        y[:K-1].Start = ybar[:K-1]
        x.Start = xbar

    if pnorm == 1:
        U = z_UB[K] - z_LB[K-1]
        L = z_LB[K] - z_UB[K-1]

        up = model.addMVar(n, ub=jnp.abs(U))
        un = model.addMVar(n, ub=jnp.abs(L))
        v = model.addMVar(n, vtype=gp.GRB.BINARY)

        for i in range(n):
            if L[i] > 0.00001:
                model.addConstr(up[i] == z[K, i] - z[K-1, i])
                model.addConstr(un[i] == 0)
            if U[i] < -0.00001:
                model.addConstr(un[i] == z[K-1, i] - z[K, i])
                model.addConstr(up[i] == 0)
            else: # Li < 0 < Ui
                model.addConstr(up[i] - un[i] == z[K, i] - z[K-1, i])
                model.addConstr(up[i] <= U[i]*v[i])
                model.addConstr(un[i] <= jnp.abs(L[i])*(1 - v[i]))

        model.setObjective(gp.quicksum(up + un), gp.GRB.MAXIMIZE)

    model.update()
    model.optimize()

    return model.objVal, model.Runtime, z.X, y.X, x.X


def VerifyPGD_withBounds_onestep(K, A, B, t, cfg, Deltas,
                                 y_LB, y_UB, z_LB, z_UB, x_LB, x_UB,
                                 zbar, ybar, xbar):

    n = cfg.n
    model = gp.Model()
    pnorm = cfg.pnorm

    var_shape = (K+1, n)
    z = model.addMVar(var_shape, lb=z_LB[:K+1], ub=z_UB[:K+1])
    x = model.addMVar(n, lb=x_LB, ub=x_UB)
    w = model.addMVar(var_shape, vtype=gp.GRB.BINARY)

    for k in range(K):
        ykplus1 = np.asarray(A) @ z[k] + np.asarray(B) @ x
        for i in range(n):
            if y_UB[k+1, i] < -0.00001:
                model.addConstr(z[k+1, i] == 0)
            elif y_LB[k+1, i] > 0.00001:
                model.addConstr(z[k+1, i] == ykplus1[i])
            else:
                model.addConstr(z[k+1, i] <= y_UB[k+1, i]/(y_UB[k+1, i] - y_LB[k+1, i]) * (ykplus1[i] - y_LB[k+1, i]))
                model.addConstr(z[k+1, i] >= ykplus1[i])
                model.addConstr(z[k+1, i] <= ykplus1[i] - y_LB[k+1, i] * (1 - w[k+1, i]))
                model.addConstr(z[k+1, i] <= y_UB[k+1, i] * w[k+1, i])

    if zbar is not None:
        z[:K-1].Start = zbar[:K-1]
        x.Start = xbar

    if pnorm == 1:
        U = z_UB[K] - z_LB[K-1]
        L = z_LB[K] - z_UB[K-1]

        up = model.addMVar(n, ub=jnp.abs(U))
        un = model.addMVar(n, ub=jnp.abs(L))
        v = model.addMVar(n, vtype=gp.GRB.BINARY)

        for i in range(n):
            if L[i] > 0.00001:
                model.addConstr(up[i] == z[K, i] - z[K-1, i])
                model.addConstr(un[i] == 0)
            if U[i] < -0.00001:
                model.addConstr(un[i] == z[K-1, i] - z[K, i])
                model.addConstr(up[i] == 0)
            else: # Li < 0 < Ui
                model.addConstr(up[i] - un[i] == z[K, i] - z[K-1, i])
                model.addConstr(up[i] <= U[i]*v[i])
                model.addConstr(un[i] <= jnp.abs(L[i])*(1 - v[i]))

        model.setObjective(gp.quicksum(up + un), gp.GRB.MAXIMIZE)

    if cfg.callback:
        model.Params.lazyConstraints = 1
        triangle_idx = []
        for j in range(n):
            if y_UB[K, j] > 0.00001 and y_LB[K, j] < 0.00001:
                triangle_idx.append(j)
        L_hatA = jnp.zeros((n, n))
        U_hatA = jnp.zeros((n, n))
        L_hatB = jnp.zeros((n, n))
        U_hatB = jnp.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if A[i, j] >= 0:
                    L_hatA = L_hatA.at[i, j].set(z_LB[K-1, j])
                    U_hatA = U_hatA.at[i, j].set(z_UB[K-1, j])
                else:
                    L_hatA = L_hatA.at[i, j].set(z_UB[K-1, j])
                    U_hatA = U_hatA.at[i, j].set(z_LB[K-1, j])

                if B[i, j] >= 0:
                    L_hatB = L_hatB.at[i, j].set(x_LB[j])
                    U_hatB = U_hatB.at[i, j].set(x_UB[j])
                else:
                    L_hatB = L_hatB.at[i, j].set(x_UB[j])
                    U_hatB = U_hatB.at[i, j].set(x_LB[j])

        A = np.asarray(A)
        B = np.asarray(B)
        L_hatA = np.asarray(L_hatA)
        U_hatA = np.asarray(U_hatA)
        L_hatB = np.asarray(L_hatB)
        U_hatB = np.asarray(U_hatB)

        def ideal_form_callback(m, where):
            if where == gp.GRB.Callback.MIPNODE: # and gp.GRB.Callback.MIPNODE_STATUS == gp.GRB.OPTIMAL:
                status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
                if status == gp.GRB.OPTIMAL:
                    zval = m.cbGetNodeRel(z)
                    xval = m.cbGetNodeRel(x)
                    wval = m.cbGetNodeRel(w)

                    for j in triangle_idx:
                        # IhatA = jnp.array(range(10))
                        # IhatA_comp = jnp.array(range(10, 20))
                        # IhatB = jnp.array(range(10))
                        # IhatB_comp = jnp.array(range(10, 20))

                        if jnp.abs(wval[K, j] - 0.5) >= 0.49:
                            continue

                        lhsA = jnp.multiply(A[j], zval[K-1])
                        rhsA = jnp.multiply(A[j], L_hatA[j] * (1 - wval[K, j]) + U_hatA[j] * wval[K, j])

                        idxA = jnp.where(lhsA < rhsA)
                        idxA_comp = jnp.where(lhsA >= rhsA)

                        lhsB = jnp.multiply(B[j], xval)
                        rhsB = jnp.multiply(B[j], L_hatB[j] * (1 - wval[K, j]) + U_hatB[j] * wval[K, j])
                        idxB = jnp.where(lhsB < rhsB)
                        idxB_comp = jnp.where(lhsB >= rhsB)

                        yA = jnp.multiply(A[j], zval[K-1] - L_hatA[j] * (1-wval[K, j]))[idxA]
                        yAcomp = jnp.multiply(A[j], U_hatA[j] * wval[K, j])[idxA_comp]
                        yB = jnp.multiply(B[j], xval - L_hatB[j] * (1-wval[K, j]))[idxB]
                        yBcomp = jnp.multiply(B[j], U_hatB[j] * wval[K, j])[idxB_comp]

                        yrhs = jnp.sum(yA) + jnp.sum(yAcomp) + jnp.sum(yB) + jnp.sum(yBcomp)
                        if zval[K, j] <= yrhs:
                            continue

                        IhatA = idxA[0]
                        IhatA_comp = idxA_comp[0]
                        IhatB = idxB[0]
                        IhatB_comp = idxB_comp[0]

                        wvar = w[K, j]
                        new_cons = 0
                        for idx in IhatA:
                            new_cons += A[j, idx] * (z[K-1, idx] - L_hatA[j, idx] * (1 - wvar))
                        for idx in IhatA_comp:
                            new_cons += A[j, idx] * U_hatA[j, idx] * wvar
                        for idx in IhatB:
                            new_cons += B[j, idx] * (x[idx] - L_hatB[j, idx] * (1 - wvar))
                        for idx in IhatB_comp:
                            new_cons += B[j, idx] * U_hatB[j, idx] * wvar

                        m.cbLazy(z[K, j].item() <= new_cons.item())

    # if cfg.callback and K == 3:
    #     model.Params.lazyConstraints = 1
    #     triangle_idx = []  # these are the indices that we actually need to bound
    #     for i in range(n):
    #         if y_UB[K, i] > 0.00001 and y_LB[K, i] < 0.00001:
    #             triangle_idx.append(i)
    #     # log.info(triangle_idx)
    #     C = jnp.hstack([A, B])
    #     L_hatC = jnp.zeros((n, 2*n))
    #     U_hatC = jnp.zeros((n, 2*n))

    #     for i in range(n):
    #         for j in range(n):
    #             if C[i, j] >= 0:
    #                 L_hatC = L_hatC.at[i, j].set(z_LB[K-1, j])
    #                 L_hatC = L_hatC.at[i, j+n].set(x_LB[j])

    #                 U_hatC = U_hatC.at[i, j].set(z_UB[K-1, j])
    #                 U_hatC = U_hatC.at[i, j+n].set(x_UB[j])
    #             else:
    #                 L_hatC = L_hatC.at[i, j].set(z_UB[K-1, j])
    #                 L_hatC = L_hatC.at[i, j+n].set(x_UB[j])

    #                 U_hatC = U_hatC.at[i, j].set(z_LB[K-1, j])
    #                 U_hatC = U_hatC.at[i, j+n].set(x_LB[j])

    #     C = np.asarray(C)
    #     L_hatC = np.asarray(L_hatC)
    #     U_hatC = np.asarray(U_hatC)

    #     def ideal_form_callback(m, where):
    #         if where == gp.GRB.Callback.MIPNODE: # and gp.GRB.Callback.MIPNODE_STATUS == gp.GRB.OPTIMAL:
    #             status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
    #             if status == gp.GRB.OPTIMAL:
    #                 zval = m.cbGetNodeRel(z)
    #                 xval = m.cbGetNodeRel(x)
    #                 wval = m.cbGetNodeRel(w)

    #                 for j in triangle_idx[1: 2]:  # only need to consider those where we cant deduce w beforehand
    #                     d = jnp.concatenate([zval[K-1], xval])
    #                     wj = wval[K, j]
    #                     lhs = jnp.multiply(C[j], d)
    #                     rhs = jnp.multiply(C[j], L_hatC[j] * (1-wj) + U_hatC[j] * wj)
    #                     Ihat = jnp.where(lhs < rhs)[0]
    #                     Ihat_comp = jnp.where(lhs >= rhs)[0]

    #                     log.info(Ihat)
    #                     log.info(Ihat_comp)

    #                     sum_Ihat = jnp.sum(jnp.multiply(C[j], d - L_hatC[j] * (1-wj))[Ihat])
    #                     sum_Ihat_comp = jnp.sum(jnp.multiply(C[j], U_hatC[j] * wj)[Ihat_comp])

    #                     sum_rhs = sum_Ihat + sum_Ihat_comp

    #                     log.info(zval[K, j])
    #                     log.info(sum_rhs)
    #                     # exit(0)

    #                     if zval[K, j] > sum_rhs:
    #                         # zx_var_stack = gp.hstack([zval[K-1, xval]])
    #                         # really have to hack this since hstack only exists in gurobi 11 which cluster does not have (yet)
    #                         new_cons = 0

    #                         for idx in Ihat:
    #                             if idx < n:
    #                                 new_cons += A[j, idx] * (z[K-1, idx] - L_hatC[j, idx] * (1 - w[K, j]))
    #                             else:
    #                                 new_cons += B[j, idx-n] * (x[idx-n] - L_hatC[j, idx] * (1 - w[K, j]))
    #                         for idx in Ihat_comp:
    #                             if idx < n:
    #                                 new_cons += A[j, idx] * U_hatC[j, idx] * w[K, j]
    #                             else:
    #                                 new_cons += B[j, idx-n] * U_hatC[j, idx] * w[K, j]
    #                         m.cbLazy(z[K, j].item() <= new_cons.item())

        model._callback = ideal_form_callback

        start = time.time()
        model.optimize(model._callback)
        manual_time = time.time() - start

    model.update()
    model.optimize()

    if cfg.callback:
        outtime = manual_time
    else:
        outtime = model.Runtime

    return model.objVal, outtime, z.X, x.X

def BoundTightY(K, A, B, t, cfg, basic=False):
    n = cfg.n
    # A = jnp.zeros((n, n))
    # B = jnp.eye(n)

    var_shape = (K+1, n)

    # First get initial lower/upper bounds with standard techniques
    y_LB = jnp.zeros(var_shape)
    y_UB = jnp.zeros(var_shape)
    z_LB = jnp.zeros(var_shape)
    z_UB = jnp.zeros(var_shape)

    # if cfg.z0.type == 'zero':
    #     z0 = jnp.zeros(n)

    if cfg.x.type == 'box':
        x_LB = cfg.x.l * jnp.ones(n)
        x_UB = cfg.x.u * jnp.ones(n)

    Bx_upper, Bx_lower = interval_bound_prop(B, x_LB, x_UB)  # only need to compute this once

    for k in range(1, K+1):
        Az_upper, Az_lower = interval_bound_prop(A, z_LB[k - 1], z_UB[k - 1])
        y_UB = y_UB.at[k].set(Az_upper + Bx_upper)
        y_LB = y_LB.at[k].set(Az_lower + Bx_lower)

        z_UB = z_UB.at[k].set(jax.nn.relu(y_UB[k]))
        z_LB = z_LB.at[k].set(jax.nn.relu(y_LB[k]))

    if basic:
        return y_LB, y_UB, z_LB, z_UB, x_LB, x_UB

    for kk in range(1, K+1):
        log.info(f'^^^^^^^^ Bound tightening, K={kk} ^^^^^^^^^^')
        for ii in range(n):
            for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
                model = gp.Model()
                model.Params.OutputFlag = 0

                z = model.addMVar(var_shape, lb=z_LB, ub=z_UB)
                y = model.addMVar(var_shape, lb=y_LB, ub=y_UB)
                x = model.addMVar(n, lb=x_LB, ub=x_UB)

                for k in range(kk):
                    model.addConstr(y[k+1] == np.asarray(A) @ z[k] + np.asarray(B) @ x)

                for k in range(kk):
                    for i in range(n):
                        if y_UB[k+1, i] < -0.00001:
                            model.addConstr(z[k+1, i] == 0)
                        elif y_LB[k+1, i] > 0.00001:
                            model.addConstr(z[k+1, i] == y[k+1, i])
                        else:
                            model.addConstr(z[k+1, i] >= y[k+1, i])
                            model.addConstr(z[k+1, i] <= y_UB[k+1, i]/ (y_UB[k+1, i] - y_LB[k+1, i]) * (y[k+1, i] - y_LB[k+1, i]))

                model.setObjective(y[kk, ii], sense)
                model.optimize()

                if model.status != gp.GRB.OPTIMAL:
                    print('bound tighting failed, GRB model status:', model.status)
                    exit(0)
                    return None

                obj = model.objVal
                if sense == gp.GRB.MAXIMIZE:
                    y_UB = y_UB.at[kk, ii].set(min(y_UB[kk, ii], obj))
                    z_UB = z_UB.at[kk, ii].set(jax.nn.relu(y_UB[kk, ii]))

                    model.setAttr(gp.GRB.Attr.UB, y[kk, ii].item(), y_UB[kk, ii])  # .item() is for MVar -> Var
                    model.setAttr(gp.GRB.Attr.UB, z[kk, ii].item(), z_UB[kk, ii])
                else:
                    y_LB = y_LB.at[kk, ii].set(max(y_LB[kk, ii], obj))
                    z_LB = z_LB.at[kk, ii].set(jax.nn.relu(y_LB[kk, ii]))

                    model.setAttr(gp.GRB.Attr.LB, y[kk, ii].item(), y_LB[kk, ii])  # .item() is for MVar -> Var
                    model.setAttr(gp.GRB.Attr.LB, z[kk, ii].item(), z_LB[kk, ii])

                model.update()

    return y_LB, y_UB, z_LB, z_UB, x_LB, x_UB


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3
    absA = jnp.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_upper, Ax_lower


def generate_P(cfg):
    n = cfg.n

    key = jax.random.PRNGKey(cfg.P_rng_seed)
    key, subkey = jax.random.split(key)
    # log.info(subkey)

    U = jax.random.orthogonal(subkey, n)
    out = jnp.zeros(n)

    key, subkey = jax.random.split(key)
    # log.info(subkey)
    out = out.at[1 : n - 1].set(
        jax.random.uniform(subkey, shape=(n - 2,), minval=cfg.mu, maxval=cfg.L)
    )

    out = out.at[0].set(cfg.mu)
    out = out.at[-1].set(cfg.L)

    if cfg.num_zero_eigvals > 0:
        out = out.at[1 : cfg.num_zero_eigvals + 1].set(0)

    P = U @ jnp.diag(out) @ U.T
    # eigs = jnp.linalg.eigvals(P)
    # log.info(f'eigval range: {jnp.min(eigs)} -> {jnp.max(eigs)}')
    # log.info(P)

    return P


def PGD(t, P, x, K):
    n = P.shape[0]
    z = jnp.zeros(n)

    for i in range(1, K+1):
        print(f'-{i}-')
        y = (jnp.eye(n) - t * P) @ z - t * x
        print('y:', y)
        znew = jax.nn.relu(y)
        print('z:', znew)
        print(jnp.linalg.norm(znew - z, 1))
        z = znew


def PGD_single(t, z, A, B, x):
    # n = A.shape[0]
    y = A @ z + B @ x
    z = jax.nn.relu(y)
    return y, z


def NNQP_run(cfg):
    log.info(cfg)
    P = generate_P(cfg)

    if cfg.stepsize.type == 'rel':
        t = cfg.stepsize.h / cfg.L
    elif cfg.stepsize.type == 'opt':
        t = 2 / (cfg.mu + cfg. L)
    elif cfg.stepsize.type == 'abs':
        t = cfg.stepsize.h

    A = jnp.eye(cfg.n) - t * P
    B = -t * jnp.eye(cfg.n)
    K_max = cfg.K_max
    K_min = cfg.K_min

    y_LB, y_UB, z_LB, z_UB, x_LB, x_UB = BoundTightY(K_max, A, B, t, cfg, basic=cfg.basic_bounding)

    Deltas = []
    solvetimes = []
    zbar_twostep, ybar, xbar_twostep = None, None, None
    for k in range(K_min, K_max + 1):
        log.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VerifyPGD_withBounds_twostep, K={k}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        delta_k, solvetime, zbar_twostep, ybar, xbar_twostep = VerifyPGD_withBounds_twostep(k, A, B, t, cfg, Deltas,
                                                                 y_LB, y_UB, z_LB, z_UB, x_LB, x_UB,
                                                                 zbar_twostep, ybar, xbar_twostep)
        # log.info(ybar)
        # log.info(zbar)
        log.info(xbar_twostep)
        Deltas.append(delta_k)
        solvetimes.append(solvetime)
        log.info(Deltas)
        log.info(solvetimes)

    Deltas_onestep = []
    solvetimes_onestep = []
    zbar, ybar, xbar = None, None, None
    for k in range(K_min, K_max + 1):
        log.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VerifyPGD_withBounds_onestep, K={k}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        delta_k, solvetime, zbar, xbar = VerifyPGD_withBounds_onestep(k, A, B, t, cfg, Deltas_onestep,
                                                                 y_LB, y_UB, z_LB, z_UB, x_LB, x_UB,
                                                                 zbar, ybar, xbar)
        # log.info(ybar)
        # log.info(zbar)
        log.info(xbar)
        Deltas_onestep.append(delta_k)
        solvetimes_onestep.append(solvetime)
        log.info(Deltas_onestep)
        log.info(solvetimes_onestep)

    log.info(f'two step deltas: {Deltas}')
    log.info(f'one step deltas: {Deltas_onestep}')
    log.info(f'two step times: {solvetimes}')
    log.info(f'one step times: {solvetimes_onestep}')

    # xbar_vec = jnp.zeros(cfg.n)
    # for i in range(cfg.n):
    #     xbar_vec = xbar_vec.at[i].set(xbar[i])

    # PGD(t, P, xbar_vec, K_max)


def run(cfg):
    NNQP_run(cfg)

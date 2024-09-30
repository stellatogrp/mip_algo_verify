import logging
import time

import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as spa

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

    if cfg.warmstart:
        if zbar is not None:
            z[:K-1].Start = zbar[:K-1]
            y[:K-1].Start = ybar[:K-1]
            x.Start = xbar

    if pnorm == 1 or pnorm == 'inf':
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

        if pnorm == 1:
            model.setObjective(cfg.obj_scaling * gp.quicksum(up + un), gp.GRB.MAXIMIZE)
        elif pnorm == 'inf':
            M = jnp.maximum(jnp.abs(U), jnp.abs(L))
            q = model.addVar(ub=jnp.max(M))
            gamma = model.addMVar(n, vtype=gp.GRB.BINARY)

            for i in range(n):
                Mi = jnp.abs(U[i]) + jnp.abs(L[i])
                model.addConstr(q >= up[i] + un[i] - Mi * (1 - gamma[i]))
                model.addConstr(q <= up[i] + un[i] + jnp.max(M) * (1 - gamma[i]))
            model.addConstr(gp.quicksum(gamma) == 1)
            model.setObjective(cfg.obj_scaling * q, gp.GRB.MAXIMIZE)
    elif pnorm == 2:
        model.setObjective(cfg.obj_scaling * (z[K] - z[K-1]) @ (z[K] - z[K-1]), gp.GRB.MAXIMIZE)

    model.update()
    model.optimize()

    if pnorm == 2:
        outobj = np.sqrt(model.objVal / cfg.obj_scaling)
    else:
        outobj = model.objVal / cfg.obj_scaling

    return outobj, model.Runtime, z.X, y.X, x.X


def computeI_Icomp(x, a, w, Lhat, Uhat):
    # lhsA = jnp.multiply(A[j], zval[k-1])
    # rhsA = jnp.multiply(A[j], L_hatA[j] * (1 - wval[k, j]) + U_hatA[j] * wval[k, j])
    lhs = jnp.multiply(a, x)
    rhs = jnp.multiply(a, Lhat * (1 - w) + Uhat * w)
    # idxA = jnp.where(lhsA < rhsA)
    # idxA_comp = jnp.where(lhsA >= rhsA)
    idx = jnp.where(lhs < rhs)
    idx_comp = jnp.where(lhs >= rhs)
    return idx, idx_comp


@jax.jit
def violation_metric(y, w, a, z, b, x, LhatA, UhatA, LhatB, UhatB):
    lhsA = jnp.multiply(a, z)
    rhsA = jnp.multiply(a, LhatA * (1 - w) + UhatA * w)
    lhsB = jnp.multiply(b, x)
    rhsB = jnp.multiply(b, LhatB * (1 - w) + UhatB * w)

    idxA = lhsA < rhsA
    idxAcomp = lhsA >= rhsA

    idxB = lhsB < rhsB
    idxBcomp = lhsB >= rhsB

    yA = jnp.where(idxA, jnp.multiply(a, z - LhatA * (1-w)), 0).sum()
    yAcomp = jnp.where(idxAcomp, jnp.multiply(a, UhatA * w), 0).sum()
    yB = jnp.where(idxB, jnp.multiply(b, x - UhatB * (1-w)), 0).sum()
    yBcomp = jnp.where(idxBcomp, jnp.multiply(b, UhatB * w), 0).sum()

    # add safety check for nan to throw an error (Warning at least)

    # return y - (yA + yAcomp + yB + yBcomp)
    return jnp.maximum(y - (yA + yAcomp + yB + yBcomp), 0)


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
        # ykplus1 = np.asarray(A) @ z[k] + np.asarray(B) @ x  # look to pass sparse scipy matrices
        ykplus1 = spa.csc_matrix(A) @ z[k] + spa.csc_matrix(B) @ x
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

    if cfg.warmstart:
        if zbar is not None:
            z[:K-1].Start = zbar[:K-1]
            x.Start = xbar

    if pnorm == 1 or pnorm == 'inf':
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
                model.addConstr(up[i] <= jnp.abs(U[i])*v[i])
                model.addConstr(un[i] <= jnp.abs(L[i])*(1 - v[i]))

        if pnorm == 1:
            model.setObjective(cfg.obj_scaling * gp.quicksum(up + un), gp.GRB.MAXIMIZE)
        elif pnorm == 'inf':
            M = jnp.maximum(jnp.abs(U), jnp.abs(L))
            q = model.addVar(ub=jnp.max(M))
            gamma = model.addMVar(n, vtype=gp.GRB.BINARY)

            for i in range(n):
                Mi = jnp.abs(U[i]) + jnp.abs(L[i])
                model.addConstr(q >= up[i] + un[i] - Mi * (1 - gamma[i]))
                model.addConstr(q <= up[i] + un[i] + jnp.max(M) * (1 - gamma[i]))
            model.addConstr(gp.quicksum(gamma) == 1)
            model.setObjective(cfg.obj_scaling * q, gp.GRB.MAXIMIZE)
    elif pnorm == 2:
        model.setObjective(cfg.obj_scaling * (z[K] - z[K-1]) @ (z[K] - z[K-1]), gp.GRB.MAXIMIZE)

    if cfg.jax_callback:
        # model.Params.lazyConstraints = 1
        model.Params.PreCrush = 1

        L_hatA = jnp.zeros((K+1, n, n))
        U_hatA = jnp.zeros((K+1, n, n))
        L_hatB = jnp.zeros((K+1, n, n))
        U_hatB = jnp.zeros((K+1, n, n))

        A_nonneg = jnp.where(A >= 0)
        A_neg = jnp.where(A < 0)
        B_nonneg = jnp.where(B >= 0)
        B_neg = jnp.where(B < 0)

        for k in range(1, K+1):
            L_hatA_k = L_hatA[k]
            L_hatA_k = L_hatA_k.at[A_nonneg].set(z_LB[k-1][A_nonneg[1]])
            L_hatA_k = L_hatA_k.at[A_neg].set(z_UB[k-1][A_neg[1]])
            L_hatA = L_hatA.at[k].set(L_hatA_k)

            U_hatA_k = U_hatA[k]
            U_hatA_k = U_hatA_k.at[A_nonneg].set(z_UB[k-1][A_nonneg[1]])
            U_hatA_k = U_hatA_k.at[A_neg].set(z_LB[k-1][A_neg[1]])
            U_hatA = U_hatA.at[k].set(U_hatA_k)

            L_hatB_k = L_hatB[k]
            L_hatB_k = L_hatB_k.at[B_nonneg].set(x_LB[B_nonneg[1]])
            L_hatB_k = L_hatB_k.at[B_neg].set(x_UB[B_neg[1]])
            L_hatB = L_hatB.at[k].set(L_hatB_k)

            U_hatB_k = U_hatB[k]
            U_hatB_k = U_hatB_k.at[B_nonneg].set(x_UB[B_nonneg[1]])
            U_hatB_k = U_hatB_k.at[B_neg].set(x_LB[B_neg[1]])
            U_hatB = U_hatB.at[k].set(U_hatB_k)

        def ideal_form_callback(m, where):
            # if where == gp.GRB.Callback.MIPNODE: # and gp.GRB.Callback.MIPNODE_STATUS == gp.GRB.OPTIMAL:
            if where == gp.GRB.Callback.MIPNODE and model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL:
                wval = jnp.asarray(m.cbGetNodeRel(w))
                # if every binary var is already 0/1, then cant cut anything so might as well exit the callback early
                if jnp.all(jnp.abs(wval - 0.5) >= cfg.binary_tol):
                    return

                nonintegral_idx = jnp.where(jnp.abs(wval - 0.5) < cfg.binary_tol)

                zval = jnp.asarray(m.cbGetNodeRel(z))
                xval = jnp.asarray(m.cbGetNodeRel(x))

                map_idx = jnp.arange(jnp.size(nonintegral_idx[0]))

                def violation_mapper(i):
                    k = nonintegral_idx[0][i]
                    j = nonintegral_idx[1][i]
                    return violation_metric(zval[k, j], wval[k, j], A[j], zval[k-1], B[j], xval,
                                            L_hatA[k, j], U_hatA[k, j], L_hatB[k, j], U_hatB[k, j])

                violations = jax.vmap(violation_mapper)(map_idx)

                if jnp.size(map_idx) <= cfg.num_top_cuts:
                    filter_idx = map_idx
                else:
                    _, filter_idx = jax.lax.top_k(violations, cfg.num_top_cuts)

                for idx in filter_idx:
                    if violations[idx] <= 0.00001:
                        continue
                    k = nonintegral_idx[0][idx]
                    j = nonintegral_idx[1][idx]
                    wvar = w[k, j]
                    IhatA, IhatA_comp = computeI_Icomp(zval[k-1], A[j], wval[k, j], L_hatA[k, j], U_hatA[k, j])
                    IhatB, IhatB_comp = computeI_Icomp(xval, B[j], wval[k, j], L_hatB[k, j], U_hatB[k, j])
                    new_cons = 0
                    for idx in IhatA[0]:
                        new_cons += A[j, idx] * (z[k-1, idx] - L_hatA[k, j, idx] * (1 - wvar))
                    for idx in IhatA_comp[0]:
                        new_cons += A[j, idx] * U_hatA[k, j, idx] * wvar
                    for idx in IhatB[0]:
                        new_cons += B[j, idx] * (x[idx] - L_hatB[k, j, idx] * (1 - wvar))
                    for idx in IhatB_comp[0]:
                        new_cons += B[j, idx] * U_hatB[k, j, idx] * wvar
                    m.cbCut(z[k, j].item() <= new_cons.item())
                    # m.cbLazy(z[k, j].item() <= new_cons.item())

        model._callback = ideal_form_callback

        start = time.time()
        model.optimize(model._callback)
        manual_time = time.time() - start


    elif cfg.callback:
        # model.Params.lazyConstraints = 1  # TODO: lazyConstraints -> PreCrush and cbLazy -> cbCut
        model.Params.PreCrush = 1
        triangle_idx = {}
        for k in range(1, K+1):
            curr_tri_idx = []
            for j in range(n):
                if y_UB[k, j] > 0.00001 and y_LB[k, j] < -0.00001:
                    curr_tri_idx.append(j)
            triangle_idx[k] = curr_tri_idx
        # log.info(triangle_idx)
        L_hatA = jnp.zeros((K+1, n, n))
        U_hatA = jnp.zeros((K+1, n, n))
        L_hatB = jnp.zeros((K+1, n, n))
        U_hatB = jnp.zeros((K+1, n, n))

        for k in range(1, K+1):  # TODO: figure out how to vectorize this
            for i in range(n):  # i is the output indices of a relu
                for j in range(n):  # j in the input indices of a relu
                    if A[i, j] >= 0:
                        L_hatA = L_hatA.at[k, i, j].set(z_LB[k-1, j])
                        U_hatA = U_hatA.at[k, i, j].set(z_UB[k-1, j])
                    else:
                        L_hatA = L_hatA.at[k, i, j].set(z_UB[k-1, j])
                        U_hatA = U_hatA.at[k, i, j].set(z_LB[k-1, j])

                    if B[i, j] >= 0:
                        L_hatB = L_hatB.at[k, i, j].set(x_LB[j])
                        U_hatB = U_hatB.at[k, i, j].set(x_UB[j])
                    else:
                        L_hatB = L_hatB.at[k, i, j].set(x_UB[j])
                        U_hatB = U_hatB.at[k, i, j].set(x_LB[j])

        A = np.asarray(A)
        B = np.asarray(B)
        allL_hatA = np.asarray(L_hatA)
        allU_hatA = np.asarray(U_hatA)
        allL_hatB = np.asarray(L_hatB)
        allU_hatB = np.asarray(U_hatB)

        def ideal_form_callback(m, where):
            if where == gp.GRB.Callback.MIPNODE: # and gp.GRB.Callback.MIPNODE_STATUS == gp.GRB.OPTIMAL:
                status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
                if status == gp.GRB.OPTIMAL:
                    zval = m.cbGetNodeRel(z)
                    xval = m.cbGetNodeRel(x)
                    wval = m.cbGetNodeRel(w)

                    # TODO: can jit the inside of this (probably)
                    # TODO: figure out the best K to use the output for
                    for k in range(max(1, K-1), K+1):
                        triangle_idxk = triangle_idx[k]
                        L_hatA = allL_hatA[k] # TODO: rename these so as to not be confusing
                        U_hatA = allU_hatA[k]
                        L_hatB = allL_hatB[k]
                        U_hatB = allU_hatB[k]
                        for j in triangle_idxk:

                            if jnp.abs(wval[k, j] - 0.5) >= 0.499:
                                continue

                            lhsA = jnp.multiply(A[j], zval[k-1])
                            rhsA = jnp.multiply(A[j], L_hatA[j] * (1 - wval[k, j]) + U_hatA[j] * wval[k, j])

                            idxA = jnp.where(lhsA < rhsA)
                            idxA_comp = jnp.where(lhsA >= rhsA)

                            lhsB = jnp.multiply(B[j], xval)
                            rhsB = jnp.multiply(B[j], L_hatB[j] * (1 - wval[k, j]) + U_hatB[j] * wval[k, j])
                            idxB = jnp.where(lhsB < rhsB)
                            idxB_comp = jnp.where(lhsB >= rhsB)

                            yA = jnp.multiply(A[j], zval[k-1] - L_hatA[j] * (1-wval[k, j]))[idxA]
                            yAcomp = jnp.multiply(A[j], U_hatA[j] * wval[k, j])[idxA_comp]
                            yB = jnp.multiply(B[j], xval - L_hatB[j] * (1-wval[k, j]))[idxB]
                            yBcomp = jnp.multiply(B[j], U_hatB[j] * wval[k, j])[idxB_comp]

                            yrhs = jnp.sum(yA) + jnp.sum(yAcomp) + jnp.sum(yB) + jnp.sum(yBcomp)
                            if zval[k, j] <= yrhs:
                                continue

                            IhatA = idxA[0]
                            IhatA_comp = idxA_comp[0]
                            IhatB = idxB[0]
                            IhatB_comp = idxB_comp[0]

                            wvar = w[k, j]
                            new_cons = 0
                            for idx in IhatA:
                                new_cons += A[j, idx] * (z[k-1, idx] - L_hatA[j, idx] * (1 - wvar))
                            for idx in IhatA_comp:
                                new_cons += A[j, idx] * U_hatA[j, idx] * wvar
                            for idx in IhatB:
                                new_cons += B[j, idx] * (x[idx] - L_hatB[j, idx] * (1 - wvar))
                            for idx in IhatB_comp:
                                new_cons += B[j, idx] * U_hatB[j, idx] * wvar

                            # m.cbLazy(z[k, j].item() <= new_cons.item())
                            m.cbCut(z[k, j].item() <= new_cons.item())

        model._callback = ideal_form_callback

        start = time.time()
        model.optimize(model._callback)
        manual_time = time.time() - start

    else:
        model.update()
        model.optimize()

    # manual_time = 0  # TODO remember to remove this once debugging is done in all K callbacks
    if cfg.callback or cfg.jax_callback:
        outtime = manual_time
    else:
        outtime = model.Runtime

    if pnorm == 2:
        outobj = np.sqrt(model.objVal / cfg.obj_scaling)
    else:
        outobj = model.objVal / cfg.obj_scaling

    return outobj, outtime, z.X, x.X

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

                    # model.setAttr(gp.GRB.Attr.UB, y[kk, ii].item(), y_UB[kk, ii])  # .item() is for MVar -> Var
                    # model.setAttr(gp.GRB.Attr.UB, z[kk, ii].item(), z_UB[kk, ii])
                else:
                    y_LB = y_LB.at[kk, ii].set(max(y_LB[kk, ii], obj))
                    z_LB = z_LB.at[kk, ii].set(jax.nn.relu(y_LB[kk, ii]))

                    # model.setAttr(gp.GRB.Attr.LB, y[kk, ii].item(), y_LB[kk, ii])  # .item() is for MVar -> Var
                    # model.setAttr(gp.GRB.Attr.LB, z[kk, ii].item(), z_LB[kk, ii])

                # model.update()

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


def jax_PGD(A, B, z0, x, K_max, pnorm=1):
    resids = jnp.zeros(K_max+1)

    if pnorm == 'inf':
        def body_fun(i, val):
            zk, resids = val
            znew = jax.nn.relu(A @ zk + B @ x)
            resids = resids.at[i].set(jnp.max(jnp.abs(znew - zk)))
            return (znew, resids)
    else:
        def body_fun(i, val):
            zk, resids = val
            znew = jax.nn.relu(A @ zk + B @ x)
            resids = resids.at[i].set(jnp.linalg.norm(znew - zk, ord=pnorm))
            return (znew, resids)

    zK, resids = jax.lax.fori_loop(1, K_max+1, body_fun, (z0, resids))
    return resids


def samples(cfg, A, B):
    sample_idx = jnp.arange(cfg.samples.N)

    def z_sample(i):
    # if cfg.z0.type == 'zero':
        # return jnp.zeros(n)
        return jnp.zeros(cfg.n)

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.n,), minval=cfg.x.l, maxval=cfg.x.u)

    z_samples = jax.vmap(z_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    def pgd_resids(i):
        return jax_PGD(A, B, z_samples[i], x_samples[i], cfg.K_max, pnorm=cfg.pnorm)

    sample_resids = jax.vmap(pgd_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


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

    max_sample_resids = samples(cfg, A, B)
    log.info(f'max sample resids: {max_sample_resids}')

    y_LB, y_UB, z_LB, z_UB, x_LB, x_UB = BoundTightY(K_max, A, B, t, cfg, basic=cfg.basic_bounding)

    # fig, ax = plt.subplots()
    if cfg.two_step:
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

            df = pd.DataFrame(Deltas)  # remove the first column of zeros
            df.to_csv(cfg.two_step_resid_fname, index=False, header=False)

            df = pd.DataFrame(solvetimes)
            df.to_csv(cfg.two_step_time_fname, index=False, header=False)

            # plotting resids so far
            fig, ax = plt.subplots()
            ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
            ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

            ax.set_xlabel(r'$K$')
            ax.set_ylabel('Fixed-point residual')
            ax.set_yscale('log')
            ax.set_title(rf'NNQP VP, $n={cfg.n}$')

            ax.legend()

            plt.savefig('twostep_resids.pdf')

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
            ax.set_title(rf'NNQP VP, $n={cfg.n}$')

            ax.legend()

            plt.savefig('twostep_times.pdf')
            plt.clf()
            plt.cla()
            plt.close()

        log.info(f'two step deltas: {Deltas}')
        log.info(f'two step times: {solvetimes}')

    if cfg.one_step:
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

            df = pd.DataFrame(Deltas_onestep)  # remove the first column of zeros
            df.to_csv(cfg.one_step_resid_fname, index=False, header=False)

            df = pd.DataFrame(solvetimes_onestep)
            df.to_csv(cfg.one_step_time_fname, index=False, header=False)

            fig, ax = plt.subplots()
            ax.plot(range(1, len(Deltas_onestep)+1), Deltas_onestep, label='VP')
            ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

            ax.set_xlabel(r'$K$')
            ax.set_ylabel('Fixed-point residual')
            ax.set_yscale('log')
            ax.set_title(rf'NNQP VP, $n={cfg.n}$')

            ax.legend()

            plt.savefig('onestep_resids.pdf')
            plt.clf()
            plt.cla()
            plt.close()

            # plotting times so far

            fig, ax = plt.subplots()
            ax.plot(range(1, len(solvetimes_onestep)+1), solvetimes_onestep, label='VP')
            # ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

            ax.set_xlabel(r'$K$')
            ax.set_ylabel('Solvetime (s)')
            ax.set_yscale('log')
            ax.set_title(rf'NNQP VP, $n={cfg.n}$')

            ax.legend()

            plt.savefig('onestep_times.pdf')

            # plt.close()
            plt.clf()
            plt.cla()
            plt.close()
        log.info(f'one step deltas: {Deltas_onestep}')
        log.info(f'one step times: {solvetimes_onestep}')

    # log.info(f'two step deltas: {Deltas}')
    # log.info(f'one step deltas: {Deltas_onestep}')
    # log.info(f'two step times: {solvetimes}')
    # log.info(f'one step times: {solvetimes_onestep}')
    # log.info(f'max sample resids: {max_sample_resids}')

    # log.info(f'infty norm: {jnp.max(jnp.abs(zbar[K_max] - zbar[K_max - 1]))}')
    # log.info(zbar[K_max] - zbar[K_max - 1])

    # xbar_vec = jnp.zeros(cfg.n)
    # for i in range(cfg.n):
    #     xbar_vec = xbar_vec.at[i].set(xbar[i])

    # PGD(t, P, xbar_vec, K_max)

    # fig, ax = plt.subplots()
    # ax.plot(range(1, len(solvetimes)+1), solvetimes, label='two step times')
    # # ax.plot(range(1, len(solvetimes_onestep)+1), solvetimes_onestep, label='one step times')
    # ax.plot(range(1, len(solvetimes_onestep)+1), solvetimes_onestep, label='one step + callback times')

    # ax.set_xlabel(r'$K$')
    # ax.set_ylabel('Solvetime(s)')
    # ax.set_yscale('log')
    # ax.set_title(r'NNQP VP, $n=20$')

    # ax.legend()

    # plt.savefig('times.pdf')


def run(cfg):
    NNQP_run(cfg)

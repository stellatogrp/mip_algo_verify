import logging

import gurobipy as gp
import jax
import jax.numpy as jnp

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

    # Variable creation for iterates/parameter
    z, y = {}, {}

    for k in range(K+1):
        for i in range(n):
            z[i, k] = model.addVar(lb=z_LB[i, k], ub=z_UB[i, k])
            if k > 0:
                y[i, k] = model.addVar(lb=y_LB[i, k], ub=y_UB[i, k])

    x = {}
    for i in range(n):
        x[i] = model.addVar(lb=x_LB[i], ub=x_UB[i])

    if pnorm == 1:
        # Variable creation for obj
        up, un, v = {}, {}, {}
        for i in range(n):
            Ui = z_UB[i,K] - z_LB[i,K-1]
            Li = z_LB[i,K] - z_UB[i,K-1]

            # should this be max of abs ?
            # Mi = jnp.abs(jnp.max(jnp.array([Ui, Li])))
            # Mi = jnp.max(jnp.abs(jnp.array([Ui, Li])))

            up[i] = model.addVar(lb=0, ub=jnp.abs(Ui))
            un[i] = model.addVar(lb=0, ub=jnp.abs(Li))

            # up[i] = model.addVar(lb=0, ub=Mi)
            # un[i] = model.addVar(lb=0, ub=Mi)

            if Li > 0.0001:
                model.addConstr(up[i] == z[i, K] - z[i, K-1])
                model.addConstr(un[i] == 0)
            elif Ui < -0.0001:
                model.addConstr(un[i] == z[i, K-1] - z[i, K])
                model.addConstr(up[i] == 0)
            else:  # Li < 0 < Ui
                v[i] = model.addVar(vtype=gp.GRB.BINARY)
                model.addConstr(up[i] - un[i] == z[i, K] - z[i, K-1])
                model.addConstr(up[i] <= Ui*v[i])
                model.addConstr(un[i] <= jnp.abs(Li)*(1 - v[i]))

    # Variable creation for MIPing relu
    w = {}
    for k in range(1, K+1):
        for i in range(n):
            w[i, k] = model.addVar(vtype=gp.GRB.BINARY)

    # Constraints for affine step
    for k in range(K):
        for i in range(n):
            model.addConstr(y[i,k+1] == gp.quicksum(A[i,j]*z[j,k] for j in range(n)) + gp.quicksum(B[i,j]*x[j] for j in range(n)))

    # Constraints for relu
    for k in range(K):
        for i in range(n):
            if y_UB[i, k+1] < -0.00001:
                model.addConstr(z[i, k+1] == 0)
            elif y_LB[i, k+1] > 0.00001:
                model.addConstr(z[i, k+1] == y[i, k+1])
            else:
                # dont need z >= 0 b/c variable bounds take care of it
                model.addConstr(z[i, k+1] <= y_UB[i,k+1]/(y_UB[i, k+1] - y_LB[i, k+1]) * (y[i, k+1] - y_LB[i, k+1]))
                model.addConstr(z[i, k+1] >= y[i, k+1])
                model.addConstr(z[i, k+1] <= y[i, k+1] - y_LB[i, k+1]*(1-w[i, k+1]))
                model.addConstr(z[i, k+1] <= y_UB[i, k+1] * w[i, k+1])

    # TODO: complete previous solution and propagate forward

    if zbar is not None:
        for i, k in zbar:
            z[i, k].Start = zbar[i,k]
        for i, k in ybar:
            y[i, k].Start = ybar[i,k]
        for i in xbar:
            x[i].Start = xbar[i]

        # this is the forward propagation, the MIP seems to run slightly slower with it
        # zKminus1 = jnp.zeros(n)
        # xKminus1 = jnp.zeros(n)
        # for i in range(n):
        #     zKminus1 = zKminus1.at[i].set(zbar[i, K-1])
        #     xKminus1 = xKminus1.at[i].set(xbar[i])
        # ynew, znew = PGD_single(t, zKminus1, A, B, xKminus1)
        # for i in range(n):
        #     y[i, K].Start = ynew[i]
        #     z[i, K].Start = znew[i]

    model.update()

    # objective formulation
    if pnorm == 1:
        model.setObjective(gp.quicksum((up[i] + un[i]) for i in range(n)), gp.GRB.MAXIMIZE)

    model.update()

    model.optimize()
    log.info(model.status)

    return model.objVal, model.Runtime, {(i,k): z[i,k].X for i, k in z}, {(i,k): y[i,k].X for i, k in y}, {j: x[j].X for j in x}


def VerifyPGD_withBounds_onestep(K, A, B, t, cfg, Deltas,
                                 y_LB, y_UB, z_LB, z_UB, x_LB, x_UB,
                                 zbar, ybar, xbar):
    n = cfg.n
    model = gp.Model()
    pnorm = cfg.pnorm

    # Variable creation for iterates/parameter
    z = {}
    for k in range(K+1):
        for i in range(n):
            z[i, k] = model.addVar(lb=z_LB[i, k], ub=z_UB[i, k])

    x = {}
    for i in range(n):
        x[i] = model.addVar(lb=x_LB[i], ub=x_UB[i])

    w = {}
    for k in range(1, K+1):
        for i in range(n):
            w[i, k] = model.addVar(vtype=gp.GRB.BINARY)

    for k in range(K):
        for i in range(n):
            if y_UB[i, k+1] < -0.00001:
                model.addConstr(z[i, k+1] == 0)
            elif y_LB[i, k+1] > 0.00001:
                z[i, k+1] = gp.quicksum(A[i,j]*z[j,k] for j in range(n)) + gp.quicksum(B[i,j]*x[j] for j in range(n))
            else:
                ykplus1 = gp.quicksum(A[i,j]*z[j,k] for j in range(n)) + gp.quicksum(B[i,j]*x[j] for j in range(n))

                # model.addConstr(z[i, k+1] <= y_UB[i,k+1]/(y_UB[i, k+1] - y_LB[i, k+1]) * (y[i, k+1] - y_LB[i, k+1]))
                # model.addConstr(z[i, k+1] >= y[i, k+1])
                # model.addConstr(z[i, k+1] <= y[i, k+1] - y_LB[i, k+1]*(1-w[i, k+1]))
                # model.addConstr(z[i, k+1] <= y_UB[i, k+1] * w[i, k+1])
                model.addConstr(z[i, k+1] <= y_UB[i,k+1]/(y_UB[i, k+1] - y_LB[i, k+1]) * (ykplus1 - y_LB[i, k+1]))
                model.addConstr(z[i, k+1] >= ykplus1)
                model.addConstr(z[i, k+1] <= ykplus1 - y_LB[i, k+1]*(1-w[i, k+1]))
                model.addConstr(z[i, k+1] <= y_UB[i, k+1] * w[i, k+1])

    if zbar is not None:
        # for i, k in zbar:
        #     z[i, k].Start = zbar[i,k]
        for i in xbar:
            x[i].Start = xbar[i]

    if pnorm == 1:
        # Variable creation for obj
        up, un, v = {}, {}, {}
        for i in range(n):
            Ui = z_UB[i,K] - z_LB[i,K-1]
            Li = z_LB[i,K] - z_UB[i,K-1]

            up[i] = model.addVar(lb=0, ub=jnp.abs(Ui))
            un[i] = model.addVar(lb=0, ub=jnp.abs(Li))

            if Li > 0.0001:
                model.addConstr(up[i] == z[i, K] - z[i, K-1])
                model.addConstr(un[i] == 0)
            elif Ui < -0.0001:
                model.addConstr(un[i] == z[i, K-1] - z[i, K])
                model.addConstr(up[i] == 0)
            else:  # Li < 0 < Ui
                v[i] = model.addVar(vtype=gp.GRB.BINARY)
                model.addConstr(up[i] - un[i] == z[i, K] - z[i, K-1])
                model.addConstr(up[i] <= Ui*v[i])
                model.addConstr(un[i] <= jnp.abs(Li)*(1 - v[i]))
        model.setObjective(gp.quicksum((up[i] + un[i]) for i in range(n)), gp.GRB.MAXIMIZE)

    model.update()

    model.optimize()
    log.info(model.status)

    zout = {}
    for k in range(K+1):
        for i in range(n):
            if isinstance(z[i, k], gp.Var):
                zout[i, k] = z[i, k].X
            else:
                zout[i, k] = z[i, k].getValue()

    # for i in range(n):
    #     log.info(f'-{i}-')
    #     log.info(f'up: {up[i].X}')
    #     log.info(f'un: {un[i].X}')
    #     log.info(f'up[i] - un[i]: {up[i].X-un[i].X}')
    #     log.info(z[i, K].getValue())
    #     if K == 1:
    #         log.info(z[i, K-1].X)
    #         log.info(z[i, K].getValue() - z[i, K-1].X)
    #     else:
    #         log.info(z[i, K-1].getValue())
    #         log.info(z[i, K].getValue() - z[i, K-1].getValue())

    return model.objVal, model.Runtime, zout, {j: x[j].X for j in x}


def BoundTightY(K, A, B, t, cfg, basic=False):
    n = cfg.n
    # A = jnp.zeros((n, n))
    # B = jnp.eye(n)

    # First get initial lower/upper bounds with standard techniques
    y_LB, y_UB = {}, {}
    z_LB, z_UB = {}, {}
    x_LB, x_UB = {}, {}

    if cfg.z0.type == 'zero':
        z0 = jnp.zeros(n)

    if cfg.x.type == 'box':
        xl = cfg.x.l * jnp.ones(n)
        xu = cfg.x.u * jnp.ones(n)

    for i in range(n):
        z_UB[i, 0] = z0[i]
        z_LB[i, 0] = z0[i]

    for i in range(n):
        x_LB[i] = xl[i]
        x_UB[i] = xu[i]

    for q in range(1, K+1):
        for i in range(n):
            y_UB[i, q]  = sum(A[i, j]*z_UB[j, q-1] for j in range(n) if A[i, j] > 0)
            y_UB[i, q] += sum(A[i, j]*z_LB[j, q-1] for j in range(n) if A[i, j] < 0)
            y_UB[i, q] += sum(B[i, j]*x_UB[j] for j in range(n) if B[i, j] > 0)
            y_UB[i, q] += sum(B[i, j]*x_LB[j] for j in range(n) if B[i, j] < 0)

            y_LB[i, q]  = sum(A[i, j]*z_LB[j, q-1] for j in range(n) if A[i, j] > 0)
            y_LB[i, q] += sum(A[i, j]*z_UB[j, q-1] for j in range(n) if A[i, j] < 0)
            y_LB[i, q] += sum(B[i, j]*x_LB[j] for j in range(n) if B[i, j] > 0)
            y_LB[i, q] += sum(B[i, j]*x_UB[j] for j in range(n) if B[i, j] < 0)

            # z_LB[i, q] = y_LB[i, q] if y_LB[i, q] > 0 else 0
            # z_UB[i, q] = y_UB[i, q] if y_UB[i, q] > 0 else 0
            z_LB[i, q] = jax.nn.relu(y_LB[i, q])
            z_UB[i, q] = jax.nn.relu(y_UB[i, q])

    if basic:
        return y_LB, y_UB, z_LB, z_UB, x_LB, x_UB

    for kk in range(1, K+1):
        log.info(f'^^^^^^^^ Bound tightening, K={kk} ^^^^^^^^^^')
        for ii in range(n):
            # have to double loop and recreate the model here because of the updating bounds
            for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
                model = gp.Model()
                model.Params.OutputFlag = 0

                z, y = {}, {}
                for k in range(K+1):
                    for i in range(n):
                        if k == 0:
                            z[i, k] = model.addVar(lb = z_LB[i, 0], ub=z_UB[i, 0])
                        else:
                            z[i, k] = model.addVar(lb=z_LB[i, k], ub=z_UB[i, k])
                            y[i, k] = model.addVar(lb=y_LB[i, k], ub=y_UB[i, k])
                x = {}
                for i in range(n):
                    x[i] = model.addVar(lb=x_LB[i], ub=x_UB[i])

                # TODO: do these need to go all the way to K? cant they stop at kk ?
                # add every affine constraint
                for k in range(kk):
                    for i in range(n):
                        model.addConstr(y[i,k+1] == gp.quicksum(A[i,j]*z[j,k] for j in range(n)) + gp.quicksum(B[i,j]*x[j] for j in range(n)))
                # constraints on relu
                for k in range(kk):
                    for i in range(n):
                        if y_UB[i, k+1] < -0.00001:
                            model.addConstr(z[i, k+1] == 0)
                        elif y_LB[i, k+1] > 0.00001:
                            model.addConstr(z[i, k+1] == y[i, k+1])
                        else:
                            model.addConstr(z[i, k+1] <= y_UB[i,k+1]/(y_UB[i, k+1] - y_LB[i, k+1]) * (y[i, k+1] - y_LB[i, k+1]))
                            model.addConstr(z[i, k+1] >= y[i, k+1])

                model.setObjective(y[ii, kk], sense)

                model.optimize()

                if model.status != gp.GRB.OPTIMAL:
                    print('bound tighting failes, GRB model status:', model.status)
                    return None

                # Update bounds
                obj = model.objVal
                if sense == gp.GRB.MAXIMIZE:
                    y_UB[ii, kk] = min(y_UB[ii, kk], obj)
                    z_UB[ii, kk] = jax.nn.relu(y_UB[ii, kk])

                    model.setAttr(gp.GRB.Attr.UB, y[ii, kk], y_UB[ii, kk])
                    model.setAttr(gp.GRB.Attr.UB, z[ii, kk], z_UB[ii, kk])
                else:
                    y_LB[ii, kk] = max(y_LB[ii, kk], obj)
                    z_LB[ii, kk] = jax.nn.relu(y_LB[ii, kk])

                    model.setAttr(gp.GRB.Attr.LB, y[ii, kk], y_LB[ii, kk])
                    model.setAttr(gp.GRB.Attr.LB, z[ii, kk], z_LB[ii, kk])

                model.update()

    return y_LB, y_UB, z_LB, z_UB, x_LB, x_UB


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

    y_LB, y_UB, z_LB, z_UB, x_LB, x_UB = BoundTightY(K_max, A, B, t, cfg, basic=cfg.basic_bounding)

    Deltas = []
    solvetimes = []
    zbar_twostep, ybar, xbar_twostep = None, None, None
    for k in range(1, K_max + 1):
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
    for k in range(1, K_max + 1):
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

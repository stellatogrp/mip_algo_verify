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
            Mi = jnp.max(jnp.abs(jnp.array([Ui, Li])))

            up[i] = model.addVar(lb=0, ub=Mi)
            un[i] = model.addVar(lb=0, ub=Mi)

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

    model.update()

    # objective formulation
    if pnorm == 1:
        model.setObjective(gp.quicksum((up[i] + un[i]) for i in range(n)), gp.GRB.MAXIMIZE)


    model.update()

    model.optimize()
    log.info(model.status)

    return model.objVal, {(i,k): z[i,k].X for i, k in z}, {(i,k): y[i,k].X for i, k in y}, {j: x[j].X for j in x}


def BoundTightY(K, A, B, t, cfg, basic=True):
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

    # for q in range(1, K+1):
    #     for i in range(n):
    #         if y_LB[i, q] < 0 and y_UB[i, q] > 0:
    #             log.info((i, q))
    #             log.info(y_LB[i, q])
    #             log.info(z_LB[i, q])
    #             log.info(y_UB[i, q])
    #             log.info(z_UB[i, q])

    if basic:
        return y_LB, y_UB, z_LB, z_UB, x_LB, x_UB

    # TODO: implement advanced version


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

    y_LB, y_UB, z_LB, z_UB, x_LB, x_UB = BoundTightY(K_max, A, B, t, cfg)

    Deltas = []
    zbar, ybar, xbar = None, None, None
    for k in range(1, K_max+1):
        log.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VerifyPGD_withBounds, K={k}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        delta_k, zbar, ybar, xbar = VerifyPGD_withBounds_twostep(k, A, B, t, cfg, Deltas,
                                                                 y_LB, y_UB, z_LB, z_UB, x_LB, x_UB,
                                                                 zbar, ybar, xbar)
        log.info(ybar)
        # log.info(zbar)
        log.info(xbar)
        Deltas.append(delta_k)
    log.info(Deltas)

    # xbar_vec = jnp.zeros(cfg.n)
    # for i in range(cfg.n):
    #     xbar_vec = xbar_vec.at[i].set(xbar[i])

    # PGD(t, P, xbar_vec, K_max)


def run(cfg):
    NNQP_run(cfg)

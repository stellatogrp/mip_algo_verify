import logging

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

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

    # log.info(vk_LB)
    # log.info(vk_UB)

    uk_LB = proj_C(cfg, vk_LB)
    uk_UB = proj_C(cfg, vk_UB)
    # log.info(uk_LB)
    # log.info(uk_UB)

    u_LB = u_LB.at[k].set(uk_LB)
    u_UB = u_LB.at[k].set(uk_UB)

    sA = jnp.block([[jnp.eye(m_plus_n), jnp.eye(m_plus_n), -jnp.eye(m_plus_n)]])
    s_rhs_l = jnp.hstack([s_LB[k-1], u_LB[k], utilde_LB[k]])
    s_rhs_u = jnp.hstack([s_UB[k-1], u_UB[k], utilde_UB[k]])
    sk_LB, sk_UB = interval_bound_prop(sA, s_rhs_l, s_rhs_u)

    # log.info(sk_LB)
    # log.info(sk_UB)

    s_LB = s_LB.at[k].set(sk_LB)
    s_UB = s_UB.at[k].set(sk_UB)

    return utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB


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
                # model.addConstr(u[k][i] >= utilde[i])
                # model.addConstr(u[k][i] <= utilde_UB[k, i] / (utilde_UB[k, i] - utilde_LB[k, i]) * (utilde[i] - utilde_LB[k, i]))
                # model.addConstr(u[k][i] <= utilde[i] - utilde_LB[k, i] * (1 - w[k, i]))
                # model.addConstr(u[k][i] <= utilde_UB[k, i] * w[k, i])
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

            for i in range(n):
                obj_constraints.append(up[i] <= np.abs(s_UB[k, i] - s_LB[k-1, i]) * vobj[i])
                obj_constraints.append(un[i] <= np.abs(s_LB[k, i] - s_UB[k-1, i]) * (1 - vobj[i]))

            for i in range(n):
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

        return model.objVal * obj_scaling, model.objBound * obj_scaling, mipgap, model.Runtime, mu.X, z_prev.X

    pnorm = cfg.pnorm
    K_max = cfg.K_max
    num_stocks, num_factors = cfg.n, cfg.d
    gamma, lambd = cfg.gamma, cfg.lambd
    m, n = A.shape

    zprev_lower = cfg.zprev.l
    zprev_upper = cfg.zprev.u

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
    for k in range(1, K_max+1):
        utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB = BoundPreprocessing(cfg, k, lhs_mat, utilde_LB, utilde_UB, v_LB, v_UB, u_LB, u_UB, s_LB, s_UB, c_l, c_u)

        # TODO: insert assertion errors

        # TODO: add theory bound

        # TODO: add obbt

        # log.info(s_LB)
        # log.info(s_UB)

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

        log.info(Deltas)
        log.info(solvetimes)
        log.info(theory_tighter_fracs)

        # TODO: postprocess


def avg_sol(cfg, D, A, b, mu):
    n, d = cfg.n, cfg.d
    z = cp.Variable(n + d)
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

    # log.info(constraints[0].dual_value)

    # return z.value
    return jnp.hstack([z.value, constraints[0].dual_value])


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

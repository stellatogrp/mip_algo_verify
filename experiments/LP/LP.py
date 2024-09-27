import logging

import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def VerifyPDHG_withBounds(K, A, c, t, cfg, Deltas,
                          utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB,
                          ubar, vbar, xbar):
    n, m = cfg.n, cfg.m

    model = gp.Model()
    pnorm = cfg.pnorm

    n_var_shape = (K+1, n)
    m_var_shape = (K+1, m)

    u = model.addMVar(n_var_shape, lb=u_LB[:K+1], ub=u_UB[:K+1])
    v = model.addMVar(m_var_shape, lb=v_LB[:K+1], ub=v_UB[:K+1])
    x = model.addMVar(m, lb=x_LB, ub=x_UB)
    w = model.addMVar(n_var_shape, vtype=gp.GRB.BINARY)

    # xC = jnp.eye(n)
    # xD = t * A.T
    # xE = - t * jnp.eye(n)

    # spa_xC = spa.csc_matrix(xC)
    # spa_xD = spa.csc_matrix(xD)
    # spa_xE = spa.csc_matrix(xE)
    # np_A = np.asarray(A)

    # vC = jnp.eye(m)
    # vD = -2 * t * A
    # vE = t * A
    # vF = t * jnp.eye(m)

    # spa_vC = spa.csc_matrix(vC)
    # spa_vD = spa.csc_matrix(vD)
    # spa_vE = spa.csc_matrix(vE)
    # spa_vF = spa.csc_matrix(vF)

    np_A = np.asarray(A)
    np_c = np.asarray(c)

    # xC = spa.eye(n)
    # xD = t * np_A.T
    # xE = -t * spa.eye(n)

    # vC = spa.eye(m)
    vD = -2 * t * np_A
    vE = t * np_A
    # vF = t * spa.eye(m)

    for k in range(K):
        model.addConstr(v[k+1] == v[k] + vD @ u[k+1] + vE @ u[k] + t * x)
        # model.addConstr(v[k+1] == v[k] - t * (np_A @ (2 * u[k+1] - u[k]) - x))

    for k in range(K):
        utildekplus1 = u[k] - t * (np_c - np_A.T @ v[k])
        for i in range(n):
            if utilde_UB[k+1, i] < -0.00001:
                model.addConstr(u[k+1, i] == 0)
            elif utilde_LB[k+1, i] > 0.00001:
                model.addConstr(u[k+1, i] == utildekplus1[i])
            else:
                model.addConstr(u[k+1, i] <= utilde_UB[k+1, i]/(utilde_UB[k+1, i] - utilde_LB[k+1, i]) * (utildekplus1[i] - utilde_LB[k+1, i]))
                model.addConstr(u[k+1, i] >= utildekplus1[i])
                model.addConstr(u[k+1, i] <= utildekplus1[i] - utilde_LB[k+1, i] * (1 - w[k+1, i]))
                model.addConstr(u[k+1, i] <= utilde_UB[k+1, i] * w[k+1, i])

    if ubar is not None:
        u[:K-1].Start = ubar[:K-1]
        v[:K-1].Start = vbar[:K-1]
        x.Start = xbar

    if pnorm == 1 or pnorm == 'inf':
        Uu = u_UB[K] - u_LB[K-1]
        Lu = u_LB[K] - u_UB[K-1]

        Uv = v_UB[K] - v_LB[K-1]
        Lv = v_LB[K] - v_UB[K-1]

        u_objp = model.addMVar(n, ub=jnp.abs(Uu))
        u_objn = model.addMVar(n, ub=jnp.abs(Lu))
        u_omega = model.addMVar(n, vtype=gp.GRB.BINARY)

        v_objp = model.addMVar(m, ub=jnp.abs(Uv))
        v_objn = model.addMVar(m, ub=jnp.abs(Lv))
        v_omega = model.addMVar(m, vtype=gp.GRB.BINARY)

        for i in range(n):
            if Lu[i] > 0.00001:
                model.addConstr(u_objp[i] == u[K, i] - u[K-1, i])
                model.addConstr(u_objn[i] == 0)
            elif Uu[i] < -0.00001:
                model.addConstr(u_objn[i] == u[K-1, i] - u[K, i])
                model.addConstr(u_objp[i] == 0)
            else:
                model.addConstr(u_objp[i] - u_objn[i] == u[K, i] - u[K-1, i])
                model.addConstr(u_objp[i] <= jnp.abs(Uu[i]) * u_omega[i])
                model.addConstr(u_objn[i] <= jnp.abs(Lu[i]) * (1-u_omega[i]))

        for i in range(m):
            if Lv[i] > 0.00001:
                model.addConstr(v_objp[i] == v[K, i] - v[K-1, i])
                model.addConstr(v_objn[i] == 0)
            elif Uv[i] < -0.00001:
                model.addConstr(v_objn[i] == v[K-1, i] - v[K, i])
                model.addConstr(v_objp[i] == 0)
            else:
                model.addConstr(v_objp[i] - v_objn[i] == v[K, i] - v[K-1, i])
                model.addConstr(v_objp[i] <= jnp.abs(Uv[i]) * v_omega[i])
                model.addConstr(v_objn[i] <= jnp.abs(Lv[i]) * (1-v_omega[i]))

        if pnorm == 1:
            model.setObjective(cfg.obj_scaling * (gp.quicksum(u_objp + u_objn) + gp.quicksum(v_objp + v_objn)), gp.GRB.MAXIMIZE)
            # model.setObjective(cfg.obj_scaling * (gp.quicksum(u_objp + u_objn)), gp.GRB.MAXIMIZE)
        elif pnorm == 'inf':
            Mu = jnp.maximum(jnp.abs(Uu), jnp.abs(Lu))
            Mv = jnp.maximum(jnp.abs(Uv), jnp.abs(Lv))
            all_max = jnp.maximum(jnp.max(Mu), jnp.max(Mv))
            q = model.addVar(ub=all_max)
            gamma_u = model.addMVar(n, vtype=gp.GRB.BINARY)
            gamma_v = model.addMVar(m, vtype=gp.GRB.BINARY)

            for i in range(n):
                Mu_i = jnp.abs(Uu[i]) + jnp.abs(Lu[i])
                model.addConstr(q >= u_objp[i] + u_objn[i] - Mu_i * (1 - gamma_u[i]))
                model.addConstr(q <= u_objp[i] + u_objn[i] + all_max * (1 - gamma_u[i]))

            for i in range(m):
                Mv_i = jnp.abs(Uv[i]) + jnp.abs(Lv[i])
                model.addConstr(q >= v_objp[i] + v_objn[i] - Mv_i * (1 - gamma_v[i]))
                model.addConstr(q <= v_objp[i] + v_objn[i] + all_max * (1 - gamma_v[i]))

            model.addConstr(gp.quicksum(gamma_u) + gp.quicksum(gamma_v) == 1)
            model.setObjective(cfg.obj_scaling * q, gp.GRB.MAXIMIZE)

    model.optimize()

    outtime = model.Runtime

    return model.objVal / cfg.obj_scaling, outtime, u.X, v.X, x.X


def BoundTight(K, A, c, t, cfg, basic=False):
    n, m = cfg.n, cfg.m

    n_var_shape = (K+1, n)
    m_var_shape = (K+1, m)

    # First get initial lower/upper bounds with standard techniques
    utilde_LB = jnp.zeros(n_var_shape)
    utilde_UB = jnp.zeros(n_var_shape)
    u_LB = jnp.zeros(n_var_shape)
    u_UB = jnp.zeros(n_var_shape)
    v_LB = jnp.zeros(m_var_shape)
    v_UB = jnp.zeros(m_var_shape)

    if cfg.x.type == 'box':
        x_LB = cfg.x.l * jnp.ones(m)
        x_UB = cfg.x.u * jnp.ones(m)

    # TODO: atm, the jax experimental sparse pacakge does not play nice with jnp.abs for bound prop
    xC = jnp.eye(n)
    xD = t * A.T
    xE = - t * jnp.eye(n)

    # spa_xC = spa.csc_matrix(xC)
    # spa_xD = spa.csc_matrix(xD)
    # spa_xE = spa.csc_matrix(xE)
    np_A = np.asarray(A)

    vC = jnp.eye(m)
    vD = -2 * t * A
    vE = t * A
    vF = t * jnp.eye(m)

    # spa_vC = spa.csc_matrix(vC)
    # spa_vD = spa.csc_matrix(vD)
    # spa_vE = spa.csc_matrix(vE)
    # spa_vF = spa.csc_matrix(vF)

    # Bx_upper, Bx_lower = interval_bound_prop(B, x_LB, x_UB)  # only need to compute this once
    vF_x_upper, vF_x_lower = interval_bound_prop(vF, x_LB, x_UB)  # only need to compute this once
    xE_c_upper, xE_c_lower = interval_bound_prop(xE, c, c)  # if c is param, change this

    for k in range(1, K+1):
        # log.info(xC)
        # log.info(u_LB[k-1])
        xC_uk_upper, xC_uk_lower = interval_bound_prop(xC, u_LB[k-1], u_UB[k-1])
        xD_vk_upper, xD_vk_lower = interval_bound_prop(xD, v_LB[k-1], v_UB[k-1])

        utilde_LB = utilde_LB.at[k].set(xC_uk_lower + xD_vk_lower + xE_c_lower)
        utilde_UB = utilde_UB.at[k].set(xC_uk_upper + xD_vk_upper + xE_c_upper)

        u_LB = u_LB.at[k].set(jax.nn.relu(utilde_LB[k]))
        u_UB = u_UB.at[k].set(jax.nn.relu(utilde_UB[k]))

        vC_vk_upper, vC_vk_lower = interval_bound_prop(vC, v_LB[k-1], v_UB[k-1])
        vD_ukplus1_upper, vD_ukplus1_lower = interval_bound_prop(vD, u_LB[k], u_UB[k])
        vE_uk_upper, vE_uk_lower = interval_bound_prop(vE, u_LB[k-1], u_UB[k-1])
        v_LB = v_LB.at[k].set(vC_vk_lower + vD_ukplus1_lower + vE_uk_lower + vF_x_lower)
        v_UB = v_UB.at[k].set(vC_vk_upper + vD_ukplus1_upper + vE_uk_upper + vF_x_upper)

    # log.info(utilde_LB)
    # log.info(utilde_UB)
    # log.info(u_LB)
    # log.info(u_UB)
    # log.info(v_LB)
    # log.info(v_UB)
    if basic:
        return utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB

    target_var = ['u_tilde', 'v']
    for kk in range(1, K+1):
        log.info(f'^^^^^^^^ Bound tightening, K={kk} ^^^^^^^^^^')
        for target in target_var:
            if target == 'u_tilde':
                range_var = n
            else:
                range_var = m
            for ii in range(range_var):
                for sense in [gp.GRB.MAXIMIZE, gp.GRB.MINIMIZE]:
                    model = gp.Model()
                    model.Params.OutputFlag = 0

                    utilde = model.addMVar(n_var_shape, lb=utilde_LB, ub=utilde_UB)
                    u = model.addMVar(n_var_shape, lb=u_LB, ub=u_UB)
                    v = model.addMVar(m_var_shape, lb=v_LB, ub=v_UB)
                    x = model.addMVar(m, lb=x_LB, ub=x_UB)

                    for k in range(kk):
                        # model.addConstr(y[k+1] == np.asarray(A) @ z[k] + np.asarray(B) @ x)
                        model.addConstr(utilde[k+1] == u[k] - t * (np.asarray(c) - np_A.T @ v[k]))
                        model.addConstr(v[k+1] == v[k] - t * (np_A @ (2 * u[k+1] - u[k]) - x))

                        for i in range(n):
                            if utilde_UB[k+1, i] < -0.00001:
                                model.addConstr(u[k+1, i] == 0)
                            elif utilde_LB[k+1, i] > 0.00001:
                                model.addConstr(u[k+1, i] == utilde[k+1, i])
                            else:
                                # model.addConstr(z[k+1, i] >= y[k+1, i])
                                # model.addConstr(z[k+1, i] <= y_UB[k+1, i]/ (y_UB[k+1, i] - y_LB[k+1, i]) * (y[k+1, i] - y_LB[k+1, i]))
                                model.addConstr(u[k+1, i] >= utilde[k+1, i])
                                model.addConstr(u[k+1, i] <= utilde_UB[k+1, i]/ (utilde_UB[k+1, i] - utilde_LB[k+1, i]) * (u[k+1, i] - utilde_LB[k+1, i]))

                    if target == 'u_tilde':
                        model.setObjective(utilde[kk, ii], sense)
                        model.optimize()
                    else:
                        model.setObjective(v[kk, ii], sense)
                        model.optimize()

                    if model.status != gp.GRB.OPTIMAL:
                        print('bound tighting failed, GRB model status:', model.status)
                        exit(0)
                        return None

                    obj = model.objVal
                    if target == 'u_tilde':
                        if sense == gp.GRB.MAXIMIZE:
                            utilde_UB = utilde_UB.at[kk, ii].set(min(utilde_UB[kk, ii], obj))
                            u_UB = u_UB.at[kk, ii].set(jax.nn.relu(utilde_UB[kk, ii]))
                        else:
                            utilde_LB = utilde_LB.at[kk, ii].set(max(utilde_LB[kk, ii], obj))
                            u_LB = u_LB.at[kk, ii].set(jax.nn.relu(utilde_LB[kk, ii]))
                    else: # target == 'v'
                        if sense == gp.GRB.MAXIMIZE:
                            v_UB = v_UB.at[kk, ii].set(min(v_UB[kk, ii], obj))
                        else:
                            v_LB = v_LB.at[kk, ii].set(max(v_LB[kk, ii], obj))

    log.info(jnp.all(utilde_UB - utilde_LB >= -1e-8))
    log.info(jnp.all(u_UB - u_LB >= -1e-8))
    log.info(jnp.all(v_UB - v_LB >= -1e-8))

    return utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB


def jax_vanilla_PDHG(A, c, t, u0, v0, x, K_max, pnorm=1):
    m, n = A.shape
    uk_all = jnp.zeros((K_max+1, n))
    vk_all = jnp.zeros((K_max+1, m))
    resids = jnp.zeros(K_max+1)

    uk_all = uk_all.at[0].set(u0)
    vk_all = vk_all.at[0].set(v0)

    def body_fun(k, val):
        uk_all, vk_all, resids = val
        uk = uk_all[k]
        vk = vk_all[k]
        ukplus1 = jax.nn.relu(uk - t * (c - A.T @ vk))
        vkplus1 = vk - t * (A @ (2 * ukplus1 - uk) - x)
        if pnorm == 'inf':
            resid = jnp.maximum(jnp.max(jnp.abs(ukplus1 - uk)), jnp.max(jnp.abs(vkplus1 - vk)))
        elif pnorm == 1:
            resid = jnp.linalg.norm(ukplus1 - uk, ord=pnorm) + jnp.linalg.norm(vkplus1 - vk, ord=pnorm)
            # resid = jnp.linalg.norm(ukplus1 - uk, ord=pnorm)
        uk_all = uk_all.at[k+1].set(ukplus1)
        vk_all = vk_all.at[k+1].set(vkplus1)
        resids = resids.at[k+1].set(resid)
        return (uk_all, vk_all, resids)

    uk, vk, resids = jax.lax.fori_loop(0, K_max, body_fun, (uk_all, vk_all, resids))
    return uk, vk, resids


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3
    absA = jnp.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_upper, Ax_lower


def samples(cfg, A, c, t):
    sample_idx = jnp.arange(cfg.samples.N)

    def u_sample(i):
        # if cfg.u0.type == 'zero':
        # return jnp.zeros(n)
        return jnp.zeros(cfg.n)

    def v_sample(i):
        # if cfg.v0.type == 'zero':
        # return jnp.zeros(m)
        return jnp.zeros(cfg.m)

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(cfg.m,), minval=cfg.x.l, maxval=cfg.x.u)

    u_samples = jax.vmap(u_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    def vanilla_pdhg_resids(i):
        return jax_vanilla_PDHG(A, c, t, u_samples[i], v_samples[i], x_samples[i], cfg.K_max, pnorm=cfg.pnorm)

    _, _, sample_resids = jax.vmap(vanilla_pdhg_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def LP_run(cfg, A, c, t):
    K_max = cfg.K_max
    K_min = cfg.K_min

    max_sample_resids = samples(cfg, A, c, t)
    log.info(max_sample_resids)

    utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB = BoundTight(K_max, A, c, t, cfg, basic=cfg.basic_bounding)

    # log.info('up right corner')
    # x = cfg.x.u * jnp.ones(cfg.m)
    # # x = jnp.array([1., 1., 0.94223])
    # uk, vk, resids = jax_vanilla_PDHG(A, c, t, jnp.zeros(cfg.n), jnp.zeros(cfg.m), x, K_max, pnorm=cfg.pnorm)
    # log.info(resids)
    # log.info(uk)
    # log.info(vk)

    # log.info('u bounds:')
    # log.info(u_LB)
    # log.info(u_UB)

    # log.info('v bounds:')
    # log.info(v_LB)
    # log.info(v_UB)

    Deltas = []
    solvetimes = []
    # zbar_twostep, ybar, xbar_twostep = None, None, None
    ubar, vbar, xbar = None, None, None
    for k in range(K_min, K_max + 1):
        log.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VerifyPDHG_with_bounds, K={k}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # def VerifyPDHG_withBounds(K, A, c, t, cfg, Deltas,
        #                   utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB,
        #                   ubar, vbar, xbar):
        delta_k, solvetime, ubar, vbar, xbar = VerifyPDHG_withBounds(k, A, c, t, cfg, Deltas,
                                                utilde_LB, utilde_UB, u_LB, u_UB, v_LB, v_UB, x_LB, x_UB,
                                                ubar, vbar, xbar)

        log.info(xbar)
        log.info(ubar)
        log.info(vbar)
        Deltas.append(delta_k)
        solvetimes.append(solvetime)
        log.info(Deltas)
        log.info(solvetimes)

        df = pd.DataFrame(Deltas)  # remove the first column of zeros
        df.to_csv(cfg.vanilla_resid_fname, index=False, header=False)

        df = pd.DataFrame(solvetimes)
        df.to_csv(cfg.vanilla_time_fname, index=False, header=False)

        # plotting resids so far
        fig, ax = plt.subplots()
        ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
        ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM')

        ax.set_xlabel(r'$K$')
        ax.set_ylabel('Fixed-point residual')
        ax.set_yscale('log')
        ax.set_title(rf'PDHG VP, $n={cfg.n}$, $m={cfg.m}$')

        ax.legend()

        plt.tight_layout()

        plt.savefig('vanilla_resids.pdf')

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
        ax.set_title(rf'PDHG VP, $n={cfg.n}$, $m={cfg.m}$')

        ax.legend()

        plt.tight_layout()

        plt.savefig('vanilla_times.pdf')
        plt.clf()
        plt.cla()
        plt.close()


def random_LP_run(cfg):
    log.info(cfg)
    m, n = cfg.m, cfg.n
    key = jax.random.PRNGKey(cfg.rng_seed)

    key, subkey = jax.random.split(key)
    A = jax.random.normal(subkey, shape=(m, n))

    key, subkey = jax.random.split(key)
    c = jax.random.uniform(subkey, shape=(n,))

    t = cfg.stepsize
    LP_run(cfg, A, c, t)


def run(cfg):
    random_LP_run(cfg)

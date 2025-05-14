import logging

import cvxpy as cp
import gurobipy as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as spa

from mipalgover.verifier import Verifier

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


def satlin(v, l, u):
    return jnp.maximum(jnp.minimum(v, u), l)


def jax_vanilla_PDHG(A_supply, A_demand, b_supply, mu, c, t, z0, v0, w0, x, K_max, pnorm=1, momentum=False, beta_func=None):
    n = mu.shape[0]
    m1 = A_supply.shape[0]
    m2 = A_demand.shape[0]

    zk_all = jnp.zeros((K_max+1, n))
    vk_all = jnp.zeros((K_max+1, m1))
    wk_all = jnp.zeros((K_max+1, m2))
    resids = jnp.zeros(K_max+1)

    zk_all = zk_all.at[0].set(z0)
    vk_all = vk_all.at[0].set(v0)
    wk_all = wk_all.at[0].set(w0)

    def body_fun(k, val):
        zk_all, vk_all, wk_all, resids = val
        zk = zk_all[k]
        vk = vk_all[k]
        wk = wk_all[k]

        zkplus1 = satlin(zk - t * (mu + A_supply.T @ vk - A_demand.T @ wk), 0, c)
        if momentum:
            ztilde_kplus1 = zkplus1 + beta_func(k) * (zkplus1 - zk)
            vkplus1 = jax.nn.relu(vk + t * (-b_supply + A_supply @ (2 * ztilde_kplus1 - zk)))
            wkplus1 = wk + t * (x - A_demand @ (2 * ztilde_kplus1 - zk))
        else:
            vkplus1 = jax.nn.relu(vk + t * (-b_supply + A_supply @ (2 * zkplus1 - zk)))
            wkplus1 = wk + t * (x - A_demand @ (2 * zkplus1 - zk))

        if pnorm == 'inf':
            resid = jnp.maximum(jnp.max(jnp.abs(zkplus1 - zk)), jnp.max(jnp.abs(vkplus1 - vk)))
            resid = jnp.maximum(resid, jnp.max(jnp.abs(wkplus1 - wk)))
        elif pnorm == 1:
            resid = jnp.linalg.norm(zkplus1 - zk, ord=pnorm) + jnp.linalg.norm(vkplus1 - vk, ord=pnorm) + jnp.linalg.norm(wkplus1 - wk, ord=pnorm)

        zk_all = zk_all.at[k+1].set(zkplus1)
        vk_all = vk_all.at[k+1].set(vkplus1)
        wk_all = wk_all.at[k+1].set(wkplus1)

        resids = resids.at[k+1].set(resid)
        return (zk_all, vk_all, wk_all, resids)

    zk, vk, wk, resids = jax.lax.fori_loop(0, K_max, body_fun, (zk_all, vk_all, wk_all, resids))
    return zk, vk, wk, resids


def get_x_LB_UB(cfg, A_demand):
    # TODO: change if we decide to use something other than a box
    m = A_demand.shape[0]
    if cfg.problem_type == 'flow':
        # b_tilde = np.hstack([b_supply, b_demand, u])

        flow = cfg.flow
        flow_x = flow.x
        demand_lb, demand_ub = flow_x.demand_lb, flow_x.demand_ub

        lb = jnp.ones(m) * demand_lb
        ub = jnp.ones(m) * demand_ub

    else:
        raise NotImplementedError

    return lb, ub


def get_capacity_vector(cfg, n):
    return jnp.ones(n) * cfg.flow.x.capacity_lb  # TODO: if change to be box


def samples(cfg, A_supply, A_demand, b_supply, mu, t, z0, v0, w0, momentum=False, beta_func=None):
    sample_idx = jnp.arange(cfg.samples.N)
    # m, n = A.shape
    n = mu.shape[0]

    def z_sample(i):
        return z0

    def v_sample(i):
        return v0

    def w_sample(i):
        return w0

    x_LB, x_UB = get_x_LB_UB(cfg, A_demand)
    log.info(x_LB)
    log.info(x_UB)
    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(A_demand.shape[0],), minval=x_LB, maxval=x_UB)

    z_samples = jax.vmap(z_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    w_samples = jax.vmap(w_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)
    c = get_capacity_vector(cfg, n)
    log.info(c)

    def vanilla_pdhg_resids(i):
        return jax_vanilla_PDHG(A_supply, A_demand, b_supply, mu, c, t, z_samples[i], v_samples[i], w_samples[i], x_samples[i], cfg.K_max,
            pnorm=cfg.pnorm, momentum=momentum, beta_func=beta_func)

    _, _, _, sample_resids = jax.vmap(vanilla_pdhg_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def max_radius(cfg, A_supply, A_demand, b_supply, mu, t, z0, v0, w0):
    n = mu.shape[0]
    m1 = v0.shape[0]
    m2 = w0.shape[0]
    c = get_capacity_vector(cfg, n)
    x_l, x_u = get_x_LB_UB(cfg, A_demand)

    model = gp.Model()

    bound_M = cfg.star_bound_M
    zstar = model.addMVar(n, lb=0, ub=c)
    vstar = model.addMVar(m1, lb=0, ub=bound_M)
    wstar = model.addMVar(m2, lb=-bound_M, ub=bound_M)
    x = model.addMVar(x_l.shape[0], lb=x_l, ub=x_u)

    model.setParam('TimeLimit', cfg.timelimit)
    model.setParam('MIPGap', cfg.mipgap)

    M = cfg.init_dist_M

    ztilde_star = zstar - t * (mu + A_supply.T @ vstar - A_demand.T @ wstar)
    vtilde_star = vstar + t * (-b_supply + A_supply @ zstar)

    z_omega1 = model.addMvar(n, vtype=gp.GRB.BINARY)
    z_omega2 = model.addMVar(n, vtype=gp.GRB.BINARY)
    v_omega = model.addMVar(m1, vtype=gp.GRB.BINARY)
    # wtilde_star = w_star + t * (x -)
    model.addConstr(x == A_demand.T @ zstar)

    for i in range(n):
        model.addConstr(zstar[i] <= c[i] * (1 - z_omega1[i]))
        model.addConstr(zstar[i] >= c[i] - c[i] * (1 - z_omega2[i]))
        model.addConstr(zstar[i] <= ztilde_star[i] + M * z_omega1[i])
        model.addConstr(zstar[i] >= ztilde_star[i] - M * z_omega2[i])

    for i in range(m1):
        model.addConstr(vstar[i] >= vtilde_star[i])
        model.addConstr(vstar[i] <= vtilde_star[i] + M * (1 - v_omega[i]))
        model.addConstr(vstar[i] <= M * v_omega[i])

    obj = (zstar - z0) @ (zstar - z0) + (vstar - v0) @ (vstar - v0) + (wstar - w0) @ (wstar - w0)
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.optimize()

    incumbent = np.sqrt(model.objVal)
    max_rad = np.sqrt(model.objBound)

    log.info(f'run time: {model.Runtime}')
    log.info(f'incumbent sol: {incumbent}')
    log.info(f'miqp max radius bound: {max_rad}')

    return incumbent, max_rad


def nesterov_beta_func(k):
    return k / (k + 3)


def LP_run(cfg, A_supply, A_demand, mu, z0):
    K = cfg.K_max
    # K_min = cfg.K_min
    momentum = cfg.momentum
    # m, n = A.shape
    # pnorm = cfg.pnorm
    m1 = A_supply.shape[0]
    m2 = A_demand.shape[0]

    if cfg.beta_func == 'nesterov':
        beta_func = nesterov_beta_func

    log.info(z0)
    log.info(f'primal dim: {z0.shape}, supply_m: {m1}, supply_m: {m2}')
    Atilde = spa.vstack([A_supply, A_demand])
    t = cfg.rel_stepsize / spa.linalg.norm(Atilde, 2)
    log.info(f'using t = {t}')

    b_supply = jnp.ones(A_supply.shape[0]) * cfg.flow.x.supply_lb  # TODO: change if parameterized

    v0 = jnp.zeros(m1)
    w0 = jnp.zeros(m2)

    A_supply = jnp.asarray(A_supply.todense())
    A_demand = jnp.asarray(A_demand.todense())

    max_sample_resids = samples(cfg, A_supply, A_demand, b_supply, mu, t, z0, v0, w0, momentum=momentum, beta_func=beta_func)
    log.info(f'max sample resids: {max_sample_resids}')

    gurobi_params = {
        'TimeLimit': cfg.timelimit,
        'MIPGap': cfg.mipgap,
    }

    VP = Verifier(solver_params=gurobi_params)

    n = mu.shape[0]
    m1 = A_supply.shape[0]
    m2 = A_demand.shape[0]
    neg_b_supply = -np.array(b_supply)

    mu_param = VP.add_param(n, lb=np.array(mu), ub=np.array(mu))
    x_l, x_u = get_x_LB_UB(cfg, A_demand)
    x_param = VP.add_param(m2, lb=np.array(x_l), ub=np.array(x_u))
    neg_b_supply_param = VP.add_param(m1, lb=neg_b_supply, ub=neg_b_supply)
    A_supply, A_demand = np.array(A_supply), np.array(A_demand)
    z0, v0, w0 = np.array(z0), np.zeros(m1), np.zeros(m2)

    z = [None for _ in range(K+1)]
    v = [None for _ in range(K+1)]
    w = [None for _ in range(K+1)]

    z[0] = VP.add_initial_iterate(n, lb=z0, ub=z0)
    v[0] = VP.add_initial_iterate(m1, lb=0, ub=0)
    w[0] = VP.add_initial_iterate(m2, lb=0, ub=0)

    Deltas = []
    rel_LP_sols = []
    Delta_bounds = []
    Delta_gaps = []
    times = []
    # theory_improv_fracs = []
    num_bin_vars = []

    relax_binary_vars = False
    c = get_capacity_vector(cfg, n)
    log.info(c)

    for k in range(1, K+1):
        log.info(f'Solving VP at k={k}')

        z[k] = VP.saturated_linear_step(z[k-1] - t * (mu_param + A_supply.T @ v[k-1] - A_demand.T @ w[k-1]), np.zeros(n), c, relax_binary_vars=relax_binary_vars)
        if momentum:
            ztilde = z[k] + beta_func(k-1) * (z[k] - z[k-1])
            v[k] = VP.relu_step(v[k-1] + t * (neg_b_supply_param + A_supply @ (2 * ztilde - z[k])), relax_binary_vars=relax_binary_vars)
            w[k] = w[k-1] + t * (x_param - A_demand @ (2 * ztilde - z[k]))
        else:
            v[k] = VP.relu_step(v[k-1] + t * (neg_b_supply_param + A_supply @ (2 * z[k] - z[k-1])), relax_binary_vars=relax_binary_vars)
            w[k] = w[k-1] + t * (x_param - A_demand @ (2 * z[k] - z[k-1]))

        VP.set_infinity_norm_objective([z[k] - z[k-1], v[k] - v[k-1], w[k] - w[k-1]])

        VP.solve(huchette_cuts=cfg.huchette_cuts, include_rel_LP_sol=False)

        data = VP.extract_solver_data()
        print(data)

        Deltas.append(data['objVal'])
        rel_LP_sols.append(data['rel_LP_sol'])
        Delta_bounds.append(data['objBound'])
        Delta_gaps.append(data['MIPGap'])
        times.append(data['Runtime'])
        num_bin_vars.append(data['numBinVars'])

        if data['Runtime'] >= cfg.relax_cutoff_time and not relax_binary_vars:
            log.info(f'FLIPPING RELAXATION AT K={k}')
            relax_binary_vars = True

        # if cfg.postprocessing:
        #     postprocess_improv = VP.post_process(z[k], z[0], np.sum(Deltas), return_improv_frac=True)
        #     log.info(f'postprocess improv: {postprocess_improv}')

        plot_data(cfg, n, m1, m2, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, num_bin_vars, times)

        print(f'samples: {max_sample_resids}')
        print(f'rel LP sols: {jnp.array(rel_LP_sols)}')
        print(f'VP residuals: {jnp.array(Deltas)}')
        print(f'VP residual bounds: {jnp.array(Delta_bounds)}')
        print(f'times:{jnp.array(times)}')


def plot_data(cfg, n, m1, m2, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, num_bin_vars, solvetimes):
    df = pd.DataFrame(Deltas)  # remove the first column of zeros
    if cfg.momentum:
        df.to_csv(cfg.momentum_resid_fname, index=False, header=False)
    else:
        df.to_csv(cfg.vanilla_resid_fname, index=False, header=False)

    df = pd.DataFrame(Delta_bounds)
    df.to_csv('resid_bounds.csv', index=False, header=False)

    df = pd.DataFrame(Delta_gaps)
    df.to_csv('resid_mip_gaps.csv', index=False, header=False)

    df = pd.DataFrame(solvetimes)
    if cfg.momentum:
        df.to_csv(cfg.momentum_time_fname, index=False, header=False)
    else:
        df.to_csv(cfg.vanilla_time_fname, index=False, header=False)

    df = pd.DataFrame(num_bin_vars)
    df.to_csv('numBinVars.csv', index=False, header=False)

    # if cfg.theory_bounds:
    #     df = pd.DataFrame(theory_tighter_fracs)
    #     df.to_csv('theory_tighter_fracs.csv', index=False, header=False)

    # plotting resids so far
    fig, ax = plt.subplots()
    ax.plot(range(1, len(Deltas)+1), Deltas, label='VP')
    ax.plot(range(1, len(rel_LP_sols)+1), rel_LP_sols, label='LP relaxations')
    ax.plot(range(1, len(Delta_bounds)+1), Delta_bounds, label='VP bounds', linewidth=5, alpha=0.3)
    ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM', linewidth=5, alpha=0.3)

    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Fixed-point residual')
    ax.set_yscale('log')
    ax.set_title(rf'PDHG VP, $n={n}$, $m_1={m1}$, $m_2 = {m2}$')

    ax.legend()

    plt.tight_layout()

    if cfg.momentum:
        plt.savefig('momentum_resids.pdf')
    else:
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
    ax.set_title(rf'PDHG VP, $n={n}$, $m_1={m1}$, $m_2 = {m2}$')

    ax.legend()

    plt.tight_layout()

    if cfg.momentum:
        plt.savefig('momentum_times.pdf')
    else:
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

    # t = cfg.stepsize
    t = cfg.rel_stepsize / jnp.linalg.norm(A, ord=2)

    if cfg.u0.type == 'zero':
        u0 = jnp.zeros(n)

    if cfg.v0.type == 'zero':
        v0 = jnp.zeros(m)

    LP_run(cfg, A, c, t, u0, v0)


def mincostflow_LP_run(cfg):
    log.info(cfg)
    flow = cfg.flow
    n_supply, n_demand, p, seed = flow.n_supply, flow.n_demand, flow.p, flow.seed

    G = nx.bipartite.random_graph(n_supply, n_demand, p, seed=seed, directed=False)
    A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)

    n_arcs = A.shape[1]
    A[n_supply:, :] *= -1

    log.info(A.todense())

    # t = cfg.rel_stepsize / spa.linalg.norm(A, ord=2)
    # log.info(f'using t={t}')

    key = jax.random.PRNGKey(flow.c.seed)
    c = jax.random.uniform(key, shape=(n_arcs,), minval=flow.c.low, maxval=flow.c.high)
    log.info(c)

    A_supply = A[:n_supply, :]
    A_demand = A[n_supply:, :]

    A_block = spa.bmat([
        [A_supply, spa.eye(n_supply), None],
        [A_demand, None, None],
        [spa.eye(n_arcs), None, spa.eye(n_arcs)]
    ])

    log.info(f'overall A size: {A_block.shape}')

    n_tilde = A_block.shape[1]
    c_tilde = np.zeros(n_tilde)
    c_tilde[:n_arcs] = c

    log.info(c_tilde)

    m, n = A_block.shape
    if flow.u0.type == 'high_demand':
        # x, _ = get_x_LB_UB(cfg, A_block)  # use the lower bound
        supply_lb = flow.x.supply_lb
        demand_lb = flow.x.demand_lb  # demand in our convention is negative so use the lower bound
        capacity_ub = flow.x.capacity_ub

        b_tilde = jnp.hstack([
            supply_lb * jnp.ones(flow.n_supply),
            demand_lb * jnp.ones(flow.n_demand),
            capacity_ub * jnp.ones(n_arcs),
        ])
        log.info(f'hardest x to satisfy: {b_tilde}')

        x_tilde = cp.Variable(n)

        constraints = [A_block @ x_tilde == b_tilde, x_tilde >= 0]

        prob = cp.Problem(cp.Minimize(c_tilde.T @ x_tilde), constraints)
        res = prob.solve(solver=cp.CLARABEL)
        log.info(res)

        if res == np.inf:
            log.info('the problem in the family with lowest supply and highest demand is infeasible')
            exit(0)

        u0 = x_tilde.value
        # v0 = constraints[0].dual_value
        v0 = jnp.zeros(m)

        log.info(f'u0: {u0}')
        log.info(f'v0: {v0}')

    else:
        if flow.u0.type == 'zero':
            u0 = jnp.zeros(n)
            # u0 = jnp.ones(n)  # TODO change back to zeros when done testing

        if flow.v0.type == 'zero':
            v0 = jnp.zeros(m)

    # LP_run(cfg, jnp.asarray(A_block.todense()), c_tilde, t, u0, v0)
    LP_run(cfg, A_supply, A_demand, c, u0[:A_supply.shape[1]])


def run(cfg):
    if cfg.problem_type == 'flow':
        mincostflow_LP_run(cfg)
    else:
        random_LP_run(cfg)

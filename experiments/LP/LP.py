import logging

import cvxpy as cp
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


def jax_vanilla_PDHG(A, c, t, u0, v0, x, K_max, pnorm=1, momentum=False, beta_func=None):
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

        if momentum:
            ytilde_kplus1 = ukplus1 + beta_func(k) * (ukplus1 - uk)
            vkplus1 = vk - t * (A @ (2 * ytilde_kplus1 - uk) - x)
        else:
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


def get_x_LB_UB(cfg, A):
    # TODO: change if we decide to use something other than a box
    if cfg.problem_type == 'flow':
        # b_tilde = np.hstack([b_supply, b_demand, u])

        flow = cfg.flow
        flow_x = flow.x

        supply_lb, supply_ub = flow_x.supply_lb, flow_x.supply_ub
        demand_lb, demand_ub = flow_x.demand_lb, flow_x.demand_ub
        capacity_lb, capacity_ub = flow_x.capacity_lb, flow_x.capacity_ub

        # log.info(A.shape)
        n_arcs = A.shape[0] - flow.n_supply - flow.n_demand
        # log.info(n_arcs)

        lb = jnp.hstack([
            supply_lb * jnp.ones(flow.n_supply),
            demand_lb * jnp.ones(flow.n_demand),
            capacity_lb * jnp.ones(n_arcs),
        ])

        ub = jnp.hstack([
            supply_ub * jnp.ones(flow.n_supply),
            demand_ub * jnp.ones(flow.n_demand),
            capacity_ub * jnp.ones(n_arcs),
        ])

        # log.info(lb)
        # log.info(ub)
    else:
        m = A.shape[0]
        lb = cfg.x.l * jnp.ones(m)
        ub = cfg.x.u * jnp.ones(m)

    return lb, ub


def samples(cfg, A, c, t, u0, v0, momentum=False, beta_func=None):
    sample_idx = jnp.arange(cfg.samples.N)
    m, n = A.shape

    def u_sample(i):
        return u0

    def v_sample(i):
        return v0

    x_LB, x_UB = get_x_LB_UB(cfg, A)
    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(m,), minval=x_LB, maxval=x_UB)

    u_samples = jax.vmap(u_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    def vanilla_pdhg_resids(i):
        return jax_vanilla_PDHG(A, c, t, u_samples[i], v_samples[i], x_samples[i], cfg.K_max, pnorm=cfg.pnorm,
                                momentum=momentum, beta_func=beta_func)

    _, _, sample_resids = jax.vmap(vanilla_pdhg_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:]


def nesterov_beta_func(k):
    return k / (k + 3)


def LP_run(cfg, A, c, t, u0, v0):
    K = cfg.K_max
    # K_min = cfg.K_min
    momentum = cfg.momentum
    m, n = A.shape
    # pnorm = cfg.pnorm

    if cfg.beta_func == 'nesterov':
        beta_func = nesterov_beta_func

    max_sample_resids = samples(cfg, A, c, t, u0, v0, momentum=momentum, beta_func=beta_func)
    log.info(max_sample_resids)

    gurobi_params = {
        'TimeLimit': cfg.timelimit,
        'MIPGap': cfg.mipgap,
    }

    VP = Verifier(solver_params=gurobi_params)

    c_param = VP.add_param(n, lb=np.array(c), ub=np.array(c))
    x_l, x_u = get_x_LB_UB(cfg, A)
    x_param = VP.add_param(m, lb=np.array(x_l), ub=np.array(x_u))
    A, u0, v0 = np.array(A), np.array(u0), np.array(v0)

    u0 = VP.add_initial_iterate(n, lb=u0, ub=u0)
    v0 = VP.add_initial_iterate(m, lb=v0, ub=v0)

    u = [None for _ in range(K+1)]
    u[0] = u0
    v = [None for _ in range(K+1)]
    v[0] = v0

    relax_cutoff = 15

    Deltas = []
    rel_LP_sols = []
    Delta_bounds = []
    Delta_gaps = []
    times = []

    for k in range(1, K+1):
        log.info(f'Solving VP at k={k}')

        u[k] = VP.relu_step(u[k-1] - t * (c_param - A.T @ v[k-1]), relax_binary_vars=(k >= relax_cutoff))

        if momentum:
            yk = u[k] + beta_func(k-1) * (u[k] - u[k-1])
            v[k] = v[k-1] - t * (A @ (2 * yk - u[k-1]) - x_param)
        else:
            v[k] = v[k-1] - t * (A @ (2 * u[k] - u[k-1]) - x_param)

        VP.set_infinity_norm_objective([u[k] - u[k-1], v[k] - v[k-1]])

        VP.solve(huchette_cuts=cfg.huchette_cuts, include_rel_LP_sol=True)

        data = VP.extract_solver_data()
        print(data)

        Deltas.append(data['objVal'])
        rel_LP_sols.append(data['rel_LP_sol'])
        Delta_bounds.append(data['objBound'])
        Delta_gaps.append(data['MIPGap'])
        times.append(data['Runtime'])

        plot_data(cfg, n, m, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, times)

        print(f'samples: {max_sample_resids}')
        print(f'rel LP sols: {jnp.array(rel_LP_sols)}')
        print(f'VP residuals: {jnp.array(Deltas)}')
        print(f'VP residual bounds: {jnp.array(Delta_bounds)}')
        print(f'times:{jnp.array(times)}')


def plot_data(cfg, n, m, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, solvetimes):
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
    ax.set_title(rf'PDHG VP, $n={n}$, $m={m}$')

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
    ax.set_title(rf'PDHG VP, $n={n}$, $m={m}$')

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

    t = cfg.rel_stepsize / spa.linalg.norm(A, ord=2)
    log.info(f'using t={t}')

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
        v0 = constraints[0].dual_value
        # v0 = jnp.zeros(m)

        log.info(f'u0: {u0}')
        log.info(f'v0: {v0}')

    else:
        if flow.u0.type == 'zero':
            u0 = jnp.zeros(n)
            # u0 = jnp.ones(n)  # TODO change back to zeros when done testing

        if flow.v0.type == 'zero':
            v0 = jnp.zeros(m)

    LP_run(cfg, jnp.asarray(A_block.todense()), c_tilde, t, u0, v0)


def run(cfg):
    if cfg.problem_type == 'flow':
        mincostflow_LP_run(cfg)
    else:
        random_LP_run(cfg)

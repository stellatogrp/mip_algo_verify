import logging

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as spa
from MPC.quadcopter import Quadcopter

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


def jax_satlin(v, l, u):
    return jnp.minimum(jnp.maximum(v, l), u)


def jax_osqp_fixedpt(K_max, P, q, A, l_rest, u_rest, xinit_samp, z0, v0, rho, sigma):
    l = jnp.hstack([xinit_samp, l_rest])
    u = jnp.hstack([xinit_samp, u_rest])

    n = z0.shape[0]
    m = v0.shape[0]

    zk_all = jnp.zeros((K_max+1, n))
    vk_all = jnp.zeros((K_max+1, m))
    resids = jnp.zeros(K_max+1)

    zk_all = zk_all.at[0].set(z0)
    vk_all = vk_all.at[0].set(v0)

    lhs_mat = P + sigma * np.eye(n) + rho * A.T @ A

    def body_fun(k, val):
        zk_all, vk_all, resids = val
        zk = zk_all[k]
        vk = vk_all[k]

        wkplus1 = jax_satlin(vk, l, u)
        zkplus1 = jnp.linalg.solve(lhs_mat, sigma * zk - q + rho * A.T @ (2 * wkplus1 - vk))
        vkplus1 = vk + A @ zkplus1 - wkplus1

        resid = jnp.maximum(jnp.max(jnp.abs(zkplus1 - zk)), jnp.max(jnp.abs(vkplus1 - vk)))

        zk_all = zk_all.at[k+1].set(zkplus1)
        vk_all = vk_all.at[k+1].set(vkplus1)

        resids = resids.at[k+1].set(resid)
        return (zk_all, vk_all, resids)

    # zk, vk, resids = jax.lax.fori_loop(0, K_max, body_fun, (zk_all, vk_all, resids))
    # log.info(resids)

    return jax.lax.fori_loop(0, K_max, body_fun, (zk_all, vk_all, resids))


def jax_osqp_nonfixedpt(K_max, P, q, A, l_rest, u_rest, xinit_samp, x0, z0, y0, rho, sigma):
    l = jnp.hstack([xinit_samp, l_rest])
    u = jnp.hstack([xinit_samp, u_rest])
    rho_inv = 1 / rho

    n = x0.shape[0]
    m = z0.shape[0]

    xk_all = jnp.zeros((K_max+1, n))
    zk_all = jnp.zeros((K_max+1, m))
    yk_all = jnp.zeros((K_max+1, m))
    resids = jnp.zeros(K_max+1)

    xk_all = xk_all.at[0].set(x0)
    zk_all = zk_all.at[0].set(z0)
    yk_all = yk_all.at[0].set(y0)

    lhs_mat = P + sigma * np.eye(n) + rho * A.T @ A

    def body_fun(k, val):
        xk_all, zk_all, yk_all, resids = val
        xk = xk_all[k]
        zk = zk_all[k]
        yk = yk_all[k]

        xkplus1 = jnp.linalg.solve(lhs_mat, sigma * xk - q + rho * A.T @ (zk - yk))
        zkplus1 = jax_satlin(A @ xkplus1 + rho_inv * yk, l, u)
        ykplus1 = yk + rho * (A @ xkplus1 - zkplus1)

        resid = jnp.maximum(jnp.max(jnp.abs(xkplus1 - xk)), jnp.max(jnp.abs(zkplus1 - zk)))
        resid = jnp.maximum(resid, jnp.max(jnp.abs(ykplus1 - yk)))

        xk_all = xk_all.at[k+1].set(xkplus1)
        zk_all = zk_all.at[k+1].set(zkplus1)
        yk_all = yk_all.at[k+1].set(ykplus1)

        resids = resids.at[k+1].set(resid)
        return (xk_all, zk_all, yk_all, resids)

    return jax.lax.fori_loop(0, K_max, body_fun, (xk_all, zk_all, yk_all, resids))


def samples(cfg, qc, P, q, A, l_rest, u_rest, xinit_l, xinit_u, z0, v0):
    sample_idx = jnp.arange(cfg.samples.N)

    def z_sample(i):
        return z0

    def v_sample(i):
        return v0

    def x_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(xinit_l.shape[0],), minval=xinit_l, maxval=xinit_u)

    z_samples = jax.vmap(z_sample)(sample_idx)
    v_samples = jax.vmap(v_sample)(sample_idx)
    x_samples = jax.vmap(x_sample)(sample_idx)

    # def vanilla_pdhg_resids(i):
    #     return jax_vanilla_PDHG(A_supply, A_demand, b_supply, mu, c, t, z_samples[i], v_samples[i], w_samples[i], x_samples[i], cfg.K_max,
    #         pnorm=cfg.pnorm, momentum=momentum, beta_func=beta_func)

    def osqp_fixedpt_resids(i):
        return jax_osqp_fixedpt(cfg.K_max, P, q, A, l_rest, u_rest, x_samples[i], z_samples[i], v_samples[i], cfg.rho, cfg.sigma)

    log.info(x_samples)

    _, _, sample_resids = jax.vmap(osqp_fixedpt_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:], x_samples


def samples_nonfixedpt(cfg, qc, P, q, A, l_rest, u_rest, xinit_l, xinit_u, x0, z0, y0):
    sample_idx = jnp.arange(cfg.samples.N)

    def x_sample(i):
        return x0

    def z_sample(i):
        return z0

    def y_sample(i):
        return y0

    def xinit_sample(i):
        key = jax.random.PRNGKey(cfg.samples.x_seed_offset + i)
        # TODO add the if, start with box case only
        return jax.random.uniform(key, shape=(xinit_l.shape[0],), minval=xinit_l, maxval=xinit_u)

    x_samples = jax.vmap(x_sample)(sample_idx)
    z_samples = jax.vmap(z_sample)(sample_idx)
    y_samples = jax.vmap(y_sample)(sample_idx)
    xinit_samples = jax.vmap(xinit_sample)(sample_idx)

    # def vanilla_pdhg_resids(i):
    #     return jax_vanilla_PDHG(A_supply, A_demand, b_supply, mu, c, t, z_samples[i], v_samples[i], w_samples[i], x_samples[i], cfg.K_max,
    #         pnorm=cfg.pnorm, momentum=momentum, beta_func=beta_func)

    def osqp_nonfixedpt_resids(i):
        return jax_osqp_nonfixedpt(cfg.K_max, P, q, A, l_rest, u_rest, xinit_samples[i], x_samples[i], z_samples[i], y_samples[i], cfg.rho, cfg.sigma)

    log.info(xinit_samples)

    _, _, _, sample_resids = jax.vmap(osqp_nonfixedpt_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:], xinit_samples


def generate_state_box(cfg, qc):
    pass


def osqp_run(cfg, qc, P, q, A, l, u):
    K = cfg.K_max
    rho, sigma = cfg.rho, cfg.sigma
    # xinit_l = qc.x0 - .1
    # xinit_u = qc.x0 + .1
    # log.info(qc.x0)
    offset = 0.1
    xinit_l = -qc.x0 - offset
    xinit_u = -qc.x0 + offset
    l_rest = l[qc.nx:]
    u_rest = u[qc.nx:]

    log.info(xinit_l)
    log.info(xinit_u)
    log.info(l_rest)
    log.info(l_rest.shape)
    log.info(u_rest)
    log.info(u_rest.shape)

    z0 = jnp.zeros(P.shape[0])
    v0 = jnp.zeros(A.shape[0])

    max_sample_resids, x_samples = samples(cfg, qc, jnp.array(P.todense()), jnp.array(q), jnp.array(A.todense()), jnp.array(l_rest), jnp.array(u_rest), jnp.array(xinit_l), jnp.array(xinit_u), z0, v0)
    log.info(max_sample_resids)

    # exit(0)

    gurobi_params = {
        'TimeLimit': cfg.timelimit,
        'MIPGap': cfg.mipgap,
    }

    VP = Verifier(solver_params=gurobi_params, obbt=False)

    q_param = VP.add_param(q.shape[0], lb=q, ub=q)

    xinit_param = VP.add_param(qc.nx, lb=xinit_l, ub=xinit_u)
    l_rest_param = VP.add_param(l_rest.shape[0], lb=l_rest, ub=l_rest)
    u_rest_param = VP.add_param(u_rest.shape[0], lb=u_rest, ub=u_rest)
    l_param = VP.add_param_stack([xinit_param, l_rest_param])
    u_param = VP.add_param_stack([xinit_param, u_rest_param])

    # l_l = np.hstack([xinit_l, l_rest])
    # l_u = np.hstack([xinit_u, l_rest])
    # u_l = np.hstack([xinit_l, u_rest])
    # u_u = np.hstack([xinit_u, u_rest])

    # l_param = VP.add_param(A.shape[0], lb=l_l, ub=l_u)
    # u_param = VP.add_param(A.shape[0], lb=u_l, ub=u_u)

    z0 = VP.add_initial_iterate(P.shape[0], lb=0, ub=0)
    v0 = VP.add_initial_iterate(A.shape[0], lb=0, ub=0)

    lhs_mat = spa.csc_matrix(P + sigma * np.eye(P.shape[0]) + rho * A.T @ A)
    lhs_factored = spa.linalg.factorized(lhs_mat)
    # lhs_mat_inv = np.asarray(np.linalg.inv(lhs_mat.todense()))

    w = [None for _ in range(K+1)]
    z = [None for _ in range(K+1)]
    v = [None for _ in range(K+1)]

    z[0] = z0
    v[0] = v0

    Deltas = []
    rel_LP_sols = []
    Delta_bounds = []
    Delta_gaps = []
    times = []
    # theory_improv_fracs = []
    num_bin_vars = []

    relax_binary_vars = False

    eq_idx_max = qc.eq_idx_max
    log.info(eq_idx_max)

    for k in range(1, K+1):
        log.info(f'Solving VP at k={k}')
        w[k] = VP.saturated_linear_param_step(v[k-1], l_param, u_param, relax_binary_vars=relax_binary_vars, equality_ranges=[(0, eq_idx_max)])
        # w[k] = VP.saturated_linear_param_step(v[k-1], l_param, u_param, relax_binary_vars=False)
        # w[k] = VP.saturated_linear_step(v[k-1], l_l, u_u, relax_binary_vars=relax_binary_vars)

        # sigma * zk - q + rho * A.T @ (2 * wkplus1 - vk)
        z[k] = VP.implicit_linear_step(lhs_mat.todense(), sigma * z[k-1] - q_param + rho * A.T @ (2 * w[k] - v[k-1]),
            lhs_mat_factorization=lhs_factored)
        # z[k] = lhs_mat_inv @ (sigma * z[k-1] - q_param + rho * A.T @ (2 * w[k] - v[k-1]))
        v[k] = v[k-1] + A @ z[k] - w[k]

        VP.set_infinity_norm_objective([z[k] - z[k-1], v[k] - v[k-1]])
        VP.solve(huchette_cuts=False, include_rel_LP_sol=False)

        data = VP.extract_solver_data()
        print(data)

        Deltas.append(data['objVal'])
        rel_LP_sols.append(data['rel_LP_sol'])
        Delta_bounds.append(data['objBound'])
        Delta_gaps.append(data['MIPGap'])
        times.append(data['Runtime'])
        num_bin_vars.append(data['numBinVars'])

        plot_data(cfg, cfg.T, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, num_bin_vars, times)

        print(f'samples: {max_sample_resids}')
        print(f'rel LP sols: {jnp.array(rel_LP_sols)}')
        print(f'VP residuals: {jnp.array(Deltas)}')
        print(f'VP residual bounds: {jnp.array(Delta_bounds)}')
        print(f'times:{jnp.array(times)}')


def osqp_nonfixedpt_run(cfg, qc, P, q, A, l, u):
    K = cfg.K_max
    rho, sigma = cfg.rho, cfg.sigma
    rho_inv = 1 / rho
    # xinit_l = qc.x0 - .1
    # xinit_u = qc.x0 + .1
    # log.info(qc.x0)
    offset = .1
    xinit_l = -qc.x0 - offset
    xinit_u = -qc.x0 + offset
    l_rest = l[qc.nx:]
    u_rest = u[qc.nx:]

    log.info(xinit_l)
    log.info(xinit_u)
    log.info(l_rest)
    log.info(l_rest.shape)
    log.info(u_rest)
    log.info(u_rest.shape)

    x0_val = qc.solve_given_x0()

    z0 = jnp.zeros(A.shape[0])
    y0 = jnp.zeros(A.shape[0])

    # max_sample_resids, x_samples = samples(cfg, qc, jnp.array(P.todense()), jnp.array(q), jnp.array(A.todense()), jnp.array(l_rest), jnp.array(u_rest), jnp.array(xinit_l), jnp.array(xinit_u), z0, v0)

    max_sample_resids, xinit_samples = samples_nonfixedpt(cfg, qc, jnp.array(P.todense()), jnp.array(q), jnp.array(A.todense()), jnp.array(l_rest), jnp.array(u_rest), jnp.array(xinit_l), jnp.array(xinit_u), jnp.array(x0_val), z0, y0)
    log.info(max_sample_resids)

    # exit(0)

    gurobi_params = {
        'TimeLimit': cfg.timelimit,
        'MIPGap': cfg.mipgap,
    }

    VP = Verifier(solver_params=gurobi_params, obbt=False)

    q_param = VP.add_param(q.shape[0], lb=q, ub=q)

    xinit_param = VP.add_param(qc.nx, lb=xinit_l, ub=xinit_u)
    l_rest_param = VP.add_param(l_rest.shape[0], lb=l_rest, ub=l_rest)
    u_rest_param = VP.add_param(u_rest.shape[0], lb=u_rest, ub=u_rest)
    l_param = VP.add_param_stack([xinit_param, l_rest_param])
    u_param = VP.add_param_stack([xinit_param, u_rest_param])

    # l_l = np.hstack([xinit_l, l_rest])
    # l_u = np.hstack([xinit_u, l_rest])
    # u_l = np.hstack([xinit_l, u_rest])
    # u_u = np.hstack([xinit_u, u_rest])

    # l_param = VP.add_param(A.shape[0], lb=l_l, ub=l_u)
    # u_param = VP.add_param(A.shape[0], lb=u_l, ub=u_u)

    x0 = VP.add_initial_iterate(P.shape[0], lb=x0_val, ub=x0_val)
    z0 = VP.add_initial_iterate(A.shape[0], lb=0, ub=0)
    y0 = VP.add_initial_iterate(A.shape[0], lb=0, ub=0)

    lhs_mat = spa.csc_matrix(P + sigma * np.eye(P.shape[0]) + rho * A.T @ A)
    lhs_factored = spa.linalg.factorized(lhs_mat)
    # lhs_mat_inv = np.asarray(np.linalg.inv(lhs_mat.todense()))

    x = [None for _ in range(K+1)]
    z = [None for _ in range(K+1)]
    y = [None for _ in range(K+1)]

    x[0] = x0
    z[0] = z0
    y[0] = y0

    Deltas = []
    rel_LP_sols = []
    Delta_bounds = []
    Delta_gaps = []
    times = []
    # theory_improv_fracs = []
    num_bin_vars = []

    # relax_binary_vars = False

    eq_idx_max = qc.eq_idx_max
    log.info(eq_idx_max)

    for k in range(1, K+1):
        log.info(f'Solving VP at k={k}')

        x[k] = VP.implicit_linear_step(lhs_mat.todense(), sigma * x[k-1] - q_param + rho * A.T @ (z[k-1] - y[k-1]),
            lhs_mat_factorization=lhs_factored)
        # x[k] = lhs_mat_inv @ (sigma * x[k-1] - q_param + rho * A.T @ (z[k-1] - y[k-1]))
        # z[k] = VP.saturated_linear_step(A @ x[k] + rho_inv * y[k-1], l_l, u_u)

        # TODO: forgot q_param, add it back
        z[k] = VP.saturated_linear_param_step(A @ x[k] + rho_inv * y[k-1], l_param, u_param, relax_binary_vars=False, equality_ranges=[(0, eq_idx_max)])
        y[k] = y[k-1] + rho * (A @ x[k] - z[k])

        VP.set_infinity_norm_objective([x[k] - x[k-1], z[k] - z[k-1], y[k] - y[k-1]])
        # VP.set_infinity_norm_objective([A @ x[k] - z[k], P @ x[k] + q_param + A.T @ y[k]])
        VP.solve(huchette_cuts=False, include_rel_LP_sol=False)

        data = VP.extract_solver_data()
        print(data)

        Deltas.append(data['objVal'])
        rel_LP_sols.append(data['rel_LP_sol'])
        Delta_bounds.append(data['objBound'])
        Delta_gaps.append(data['MIPGap'])
        times.append(data['Runtime'])
        num_bin_vars.append(data['numBinVars'])

        plot_data(cfg, cfg.T, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, num_bin_vars, times)

        print(f'samples: {max_sample_resids}')
        print(f'rel LP sols: {jnp.array(rel_LP_sols)}')
        print(f'VP residuals: {jnp.array(Deltas)}')
        print(f'VP residual bounds: {jnp.array(Delta_bounds)}')
        print(f'times:{jnp.array(times)}')


def plot_data(cfg, T, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, num_bin_vars, solvetimes):
    df = pd.DataFrame(Deltas)  # remove the first column of zeros
    df.to_csv('resids.csv', index=False, header=False)

    df = pd.DataFrame(Delta_bounds)
    df.to_csv('resid_bounds.csv', index=False, header=False)

    df = pd.DataFrame(Delta_gaps)
    df.to_csv('resid_mip_gaps.csv', index=False, header=False)

    df = pd.DataFrame(solvetimes)
    df.to_csv('solvetimes.csv', index=False, header=False)

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
    ax.set_title(rf'MPC VP, $T={T}$')

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
    ax.set_title(rf'MPC VP, $T={T}$')

    ax.legend()

    plt.tight_layout()
    plt.savefig('solvetimes.pdf')
    plt.clf()
    plt.cla()
    plt.close()


def mpc_run(cfg):
    qc = Quadcopter(T=cfg.T)
    P, q, A, l, u = qc.P, qc.q, qc.A, qc.l, qc.u

    log.info(f'P shape: {P.shape}')
    log.info(f'A shape: {A.shape}')
    log.info(q)

    # lhs_mat = P + sigma * np.eye(n) + rho * A.T @ A
    # wkplus1 = proj(vk, l, u)
    # xkplus1 = np.linalg.solve(lhs_mat, sigma * xk - q + rho * A.T @ (2 * wkplus1 - vk))
    # vkplus1 = vk + A @ xkplus1 - wkplus1

    # osqp_run(cfg, qc, P, q, A, l, u)
    osqp_nonfixedpt_run(cfg, qc, P, q, A, l, u)


def run(cfg):
    log.info(cfg)
    mpc_run(cfg)

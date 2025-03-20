import logging

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as spa

# from MPC.quadcopter import Quadcopter
from MPC.quadcopter_compact import Quadcopter

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


def jax_osqp_fixedpt(K_max, P, B, A, l, u, xinit_samp, z0, v0, rho, sigma):
    # l = jnp.hstack([xinit_samp, l_rest])
    # u = jnp.hstack([xinit_samp, u_rest])

    n = z0.shape[0]
    m = v0.shape[0]

    zk_all = jnp.zeros((K_max+1, n))
    vk_all = jnp.zeros((K_max+1, m))
    resids = jnp.zeros(K_max+1)

    zk_all = zk_all.at[0].set(z0)
    vk_all = vk_all.at[0].set(v0)

    lhs_mat = P + sigma * np.eye(n) + A.T @ rho @ A

    def body_fun(k, val):
        zk_all, vk_all, resids = val
        zk = zk_all[k]
        vk = vk_all[k]

        wkplus1 = jax_satlin(vk, l, u)
        zkplus1 = jnp.linalg.solve(lhs_mat, sigma * zk - B @ xinit_samp + A.T @ rho @ (2 * wkplus1 - vk))
        vkplus1 = vk + A @ zkplus1 - wkplus1

        resid = jnp.maximum(jnp.max(jnp.abs(zkplus1 - zk)), jnp.max(jnp.abs(vkplus1 - vk)))

        zk_all = zk_all.at[k+1].set(zkplus1)
        vk_all = vk_all.at[k+1].set(vkplus1)

        resids = resids.at[k+1].set(resid)
        return (zk_all, vk_all, resids)

    # zk, vk, resids = jax.lax.fori_loop(0, K_max, body_fun, (zk_all, vk_all, resids))
    # log.info(resids)

    return jax.lax.fori_loop(0, K_max, body_fun, (zk_all, vk_all, resids))

def samples(cfg, qc, P, B, A, l_rest, u_rest, xinit_l, xinit_u, z0, v0):
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

    rho = build_rho(cfg, cfg.nx, P.shape[0])

    def osqp_fixedpt_resids(i):
        return jax_osqp_fixedpt(cfg.K_max, P, B, A, l_rest, u_rest, x_samples[i], z_samples[i], v_samples[i], jnp.array(rho.todense()), cfg.sigma)

    log.info(x_samples)

    _, _, sample_resids = jax.vmap(osqp_fixedpt_resids)(sample_idx)
    log.info(sample_resids)
    max_sample_resids = jnp.max(sample_resids, axis=0)
    log.info(max_sample_resids)

    df = pd.DataFrame(sample_resids[:, 1:])  # remove the first column of zeros
    df.to_csv(cfg.samples.out_fname, index=False, header=False)

    return max_sample_resids[1:], x_samples


def build_rho(cfg, nx, n):
    if cfg.rho_type == 'scalar':
        return cfg.rho * spa.eye(n)
    else:
        rho = cfg.rho * np.ones(n)
        rho[:nx] *= cfg.rho_eq_scalar
        return spa.diags_array(rho)


def theory_func(k):
        # if k == 1:
        #     return np.inf
        # return 2 * init_C / np.sqrt((k-1) * (k+2))

        return np.sqrt(9 ** 2 / k + 1)


def osqp_run(cfg, qc, P, q, A, l, u, x_ws):
    K = cfg.K_max
    # rho, sigma = cfg.rho, cfg.sigma
    sigma = cfg.sigma
    rho = build_rho(cfg, cfg.nx, P.shape[0])

    # xinit = qc.xinit
    # offset = 0.1
    # xinit_l = xinit - offset
    # xinit_u = xinit + offset

    xinit_l = np.array([0., 0., 1., 0.,0.,0.,0.,0.,0.,0.,0.,0.])
    xinit_u = np.array([np.pi/6,np.pi/6, 1., 0.,0.,0.,0.,0.,0.,0.,0.,0.])

    # z0 = jnp.zeros(P.shape[0])
    z0 = jnp.array(x_ws)
    v0 = jnp.zeros(A.shape[0])

    B = qc.no_xinit_q_multipler
    log.info(B.shape)

    max_sample_resids, x_samples = samples(cfg, qc, jnp.array(P.todense()), jnp.array(B.todense()), jnp.array(A.todense()), jnp.array(l), jnp.array(u), jnp.array(xinit_l), jnp.array(xinit_u), z0, v0)
    log.info(max_sample_resids)

    gurobi_params = {
        'TimeLimit': cfg.timelimit,
        'MIPGap': cfg.mipgap,
        # 'MIPFocus': 3,
    }

    VP = Verifier(solver_params=gurobi_params, obbt=True, theory_func=theory_func)

    # q_param = VP.add_param(q.shape[0], lb=q, ub=q)

    # xinit_param = VP.add_param(qc.nx, lb=xinit_l, ub=xinit_u)
    # l_rest_param = VP.add_param(l_rest.shape[0], lb=l_rest, ub=l_rest)
    # u_rest_param = VP.add_param(u_rest.shape[0], lb=u_rest, ub=u_rest)
    # l_param = VP.add_param_stack([xinit_param, l_rest_param])

    xinit_param = VP.add_param(B.shape[1], lb=xinit_l, ub=xinit_u)
    # z0 = VP.add_initial_iterate(P.shape[0], lb=0, ub=0)
    z0 = VP.add_initial_iterate(P.shape[0], lb=x_ws, ub=x_ws)
    v0 = VP.add_initial_iterate(A.shape[0], lb=0, ub=0)

    # lhs_mat = spa.csc_matrix(P + sigma * np.eye(P.shape[0]) + rho * A.T @ A)
    lhs_mat = spa.csc_matrix(P + sigma * np.eye(P.shape[0]) + A.T @ rho @ A)
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
    theory_improv_fracs = []
    num_bin_vars = []

    relax_binary_vars = False

    eq_idx_max = qc.nx
    log.info(eq_idx_max)

    for k in range(1, K+1):
        log.info(f'Solving VP at k={k}')
        # w[k] = VP.saturated_linear_param_step(v[k-1], l_param, u_param, relax_binary_vars=relax_binary_vars, equality_ranges=[(0, eq_idx_max)])
        w[k] = VP.saturated_linear_step(v[k-1], l, u, relax_binary_vars=relax_binary_vars)

        z[k] = VP.implicit_linear_step(lhs_mat.todense(), sigma * z[k-1] - B @ xinit_param + A.T @ rho @ (2 * w[k] - v[k-1]),
            lhs_mat_factorization=lhs_factored)
        # z[k] = lhs_mat_inv @ (sigma * z[k-1] - B @ xinit_param + A.T @ rho @ (2 * w[k] - v[k-1]))

        # theory_improv = VP.theory_bound(k, z[k], z[k-1])
        v[k] = v[k-1] + A @ z[k] - w[k]

        VP.set_infinity_norm_objective([z[k] - z[k-1], v[k] - v[k-1]])
        VP.solve(huchette_cuts=True, include_rel_LP_sol=False)

        data = VP.extract_solver_data()
        print(data)

        Deltas.append(data['objVal'])
        rel_LP_sols.append(data['rel_LP_sol'])
        Delta_bounds.append(data['objBound'])
        Delta_gaps.append(data['MIPGap'])
        times.append(data['Runtime'])
        num_bin_vars.append(data['numBinVars'])
        # theory_improv_fracs.append(theory_improv)

        plot_data(cfg, cfg.T, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, num_bin_vars, times)

        print(f'samples: {max_sample_resids}')
        print(f'rel LP sols: {jnp.array(rel_LP_sols)}')
        print(f'VP residuals: {jnp.array(Deltas)}')
        print(f'VP residual bounds: {jnp.array(Delta_bounds)}')
        print(f'theory improv fracs: {jnp.array(theory_improv_fracs)}')
        print(f'times:{jnp.array(times)}')

    # xinit_vp = VP.extract_sol(xinit_param)
    # log.info(f'xinit_vp: {xinit_vp}')
    # _, _, resids = jax_osqp_fixedpt(cfg.K_max, jnp.array(P.todense()), jnp.array(q), jnp.array(A.todense()), jnp.array(l_rest), jnp.array(u_rest), xinit_vp, jnp.zeros(P.shape[0]), jnp.zeros(A.shape[0]), cfg.rho, cfg.sigma)
    # log.info(resids)

    log.info(VP.extract_sol(z[k]))


def plot_data(cfg, T, max_sample_resids, Deltas, rel_LP_sols, Delta_bounds, Delta_gaps, num_bin_vars, solvetimes, plot=False):
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

    df = pd.DataFrame(max_sample_resids)
    df.to_csv('max_sample_resids.csv', index=False, header=False)

    # if cfg.theory_bounds:
    #     df = pd.DataFrame(theory_tighter_fracs)
    #     df.to_csv('theory_tighter_fracs.csv', index=False, header=False)

    if not plot:
        return

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
    # P, q, A, l, u = qc.P, qc.q, qc.A, qc.l, qc.u
    # P, q, A, l, u, x_test = qc.test_simplified_cvxpy()
    P, q, A, l, u, x_test = qc.test_cvxpy_no_xinit()

    log.info(f'P shape: {P.shape}')
    log.info(f'A shape: {A.shape}')
    log.info(q)

    eigs, _ = spa.linalg.eigs(P)
    log.info(np.real(eigs))

    # lhs_mat = P + sigma * np.eye(n) + rho * A.T @ A
    # wkplus1 = proj(vk, l, u)
    # xkplus1 = np.linalg.solve(lhs_mat, sigma * xk - q + rho * A.T @ (2 * wkplus1 - vk))
    # vkplus1 = vk + A @ xkplus1 - wkplus1

    osqp_run(cfg, qc, P, q, A, l, u, x_test)


def run(cfg):
    log.info(cfg)
    mpc_run(cfg)

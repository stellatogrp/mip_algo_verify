import cvxpy as cp
import numpy as np

from mipalgover.verifier import Verifier

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_satlin():
    np.random.seed(8)

    n = 3

    # P = np.random.normal(size=(n, n))
    # P = P @ P.T
    P = np.eye(n)
    # q = np.random.normal(size=(n,))
    q = np.zeros(n)

    # l = 0.5 * np.random.normal(size=(n,))
    # u = l + 0.1
    l = np.array([-1, 1, -0.1])
    u = np.array([-0.9, 1.01, 0.1])

    x = cp.Variable(n)
    obj = .5 * cp.quad_form(x, P) + q.T @ x
    constraints = [x >= l, x <=u]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve()

    print('test solution from cvxpy:', x.value)

    K = 100
    t = .01
    z0_init = (l + u) / 2
    zk = z0_init
    for _ in range(K):
        yk = zk - t * (P @ zk + q)
        zk = proj(yk, l, u)

    print('solution from pgd:', zk)
    assert np.linalg.norm(x.value - zk) <= 1e-7

    gurobi_params = {
        'TimeLimit': 10,
        'MIPGap': 1e-4,
    }
    VP = Verifier(solver_params=gurobi_params)
    q_offset = 1
    q_param = VP.add_param(n, lb=q-q_offset, ub=q+q_offset)

    z0 = VP.add_initial_iterate(n, lb=z0_init, ub=z0_init)

    K = 12
    z = [None for _ in range(K+1)]
    z[0] = z0

    all_res = []
    for k in range(1, K+1):
        print(k)
        z[k] = VP.saturated_linear_step(z[k-1] - t * (P @ z[k-1] + q_param), l, u)

        VP.set_infinity_norm_objective(z[k] - z[k-1])
        res = VP.solve()
        all_res.append(res)

    print(f'opt q_param at last K: {VP.extract_sol(q_param)}')
    print(f'VP resids: {all_res}')

    print('extracting values from VP:')
    for k in range(K+1):
        print(f'-K={k}-')
        print(VP.extract_sol(z[k]))

    z_test = np.array([-0.91651, 1, 0])
    print('test grad step from K-1:', z_test - t * (P @ z_test + VP.extract_sol(q_param)))
    print('bounds:', VP.lower_bounds[VP.iterates[-1]], VP.upper_bounds[VP.iterates[-1]])

    zk = z0_init
    q = VP.extract_sol(q_param)
    PGD_resids = []

    # print('z0_init:', z0_init)
    for k in range(K):
        znew = proj(zk - t * (P @ zk + q), l, u)
        # print('test zk:', znew)
        PGD_resids.append(np.max(np.abs(znew - zk)))
        zk = znew

    print('PGD resids with last K:', PGD_resids)
    assert np.all(np.array(PGD_resids) <= np.array(all_res) + 1e-7)
    assert np.abs(PGD_resids[-1] - all_res[-1]) <= 1e-6


def proj(v, l, u):
    return np.minimum(np.maximum(v, l), u)

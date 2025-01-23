import cvxpy as cp
import numpy as np

from mipalgover.verifier import Verifier

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_partial_nnqp():
    np.random.seed(0)
    n = 3
    proj_ranges = (1, 3)

    P = np.random.normal(size=(n, n))
    P = P @ P.T
    q = np.random.normal(size=(n,))

    # testing cvxpy solution
    x = cp.Variable(n)
    obj = .5 * cp.quad_form(x, P) + q.T @ x
    constraints = [x[proj_ranges[0]: proj_ranges[1]] >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    prob.solve()
    print('test solution from cp:', x.value)

    K = 1000
    t = .01
    zk = -np.ones(n)
    for _ in range(K):
        # yk = (np.eye(n) - t * P) @ zk - t * q
        yk = zk - t * (P @ zk + q)
        zk = partial_relu(yk, proj_ranges)

    print('solution from pgd:', zk)

    assert np.linalg.norm(x.value - zk) <= 1e-7

    VP = Verifier()

    q_offset = 0.1
    q_param = VP.add_param(n, lb=q-q_offset, ub=q+q_offset)

    z0 = VP.add_initial_iterate(n, lb=-1, ub=-1)
    K = 10
    z = [None for _ in range(K+1)]
    z[0] = z0

    all_res = []
    for k in range(1, K+1):
        print(k)
        z[k] = VP.relu_step(z[k-1] - t * (P @ z[k-1] + q_param), proj_ranges=proj_ranges)

        VP.set_infinity_norm_objective(z[k] - z[k-1])
        res = VP.solve(full_convexify=True)
        all_res.append(res)

    print(f'opt q_param at last K: {VP.extract_sol(q_param)}')

    print(f'VP resids: {all_res}')

    zk = -np.ones(n)
    q = VP.extract_sol(q_param)
    PGD_resids = []

    for k in range(K):
        znew = partial_relu(zk - t * (P @ zk + q), proj_ranges)
        PGD_resids.append(np.max(np.abs(znew - zk)))
        zk = znew

    print(PGD_resids)

    assert np.all(np.array(PGD_resids) <= np.array(all_res) + 1e-7)
    assert np.abs(PGD_resids[-1] - all_res[-1]) <= 1e-8


def partial_relu(x, proj_ranges):
    out = x.copy()
    out[proj_ranges[0]: proj_ranges[1]] = np.maximum(x[proj_ranges[0]: proj_ranges[1]], 0)
    return out

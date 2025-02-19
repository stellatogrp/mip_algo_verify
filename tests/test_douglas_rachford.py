import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from mipalgover.verifier import Verifier

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_dr():
    np.random.seed(5)

    n = 4
    m = 3

    Phalf = np.random.normal(size=(n, n))
    P = Phalf @ Phalf.T + 0.1 * np.eye(n)
    c = np.random.normal(size=n)

    print(np.linalg.eigvals(P))

    A = np.random.normal(size=(m, n))
    xtest = 1/n * np.random.normal(size=n)
    b = A @ xtest

    print('--testing with cvxpy--')
    x = cp.Variable(n)
    s = cp.Variable(m)
    obj = .5 * cp.quad_form(x, P) + c.T @ x
    constraints = [A @ x + s == b, s >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    print('optimal x from cp:', x.value)
    print('optimal s from cp:', s.value)
    print('optimal y from cp:', constraints[1].dual_value)

    print('testing with DR splitting:')
    M = np.block([
        [P, A.T],
        [-A, np.zeros((m, m))]
    ])
    lhs = spa.csc_matrix(M) + spa.eye(m + n)
    lhs_factored = spa.linalg.factorized(lhs)
    q = np.hstack([c, b])

    zk = np.zeros(m + n)

    K = 1000
    for _ in range(K):
        u = spa.linalg.spsolve(lhs, zk - q)
        utilde = proj(2 * u - zk, n)
        zk = zk + utilde - u

    print('z from DR:', zk)

    VP = Verifier()

    c_offset = 0.1
    q_l = np.hstack([c - c_offset, b])
    q_u = np.hstack([c + c_offset, b])
    q_param = VP.add_param(n + m, lb=q_l, ub=q_u)

    z0_init = np.zeros(n + m)
    z0 = VP.add_initial_iterate(m + n, lb=z0_init, ub=z0_init)
    K = 5
    u = [None for _ in range(K+1)]
    utilde = [None for _ in range(K+1)]
    z = [None for _ in range(K+1)]
    z[0] = z0

    all_res = []

    # VP = Verifier()
    for k in range(1, K+1):
        print(f'-K = {k}-')
        u[k] = VP.implicit_linear_step(lhs.todense(), z[k-1] - q_param,
            lhs_mat_factorization=lhs_factored)
        utilde[k] = VP.relu_step(2 * u[k] - z[k-1], proj_ranges=(n, n+m))
        z[k] = z[k-1] + utilde[k] - u[k]

        VP.set_infinity_norm_objective(z[k] - z[k-1])
        res = VP.solve()
        all_res.append(res)

    print(f'opt q_param at last K: {VP.extract_sol(q_param)}')
    print(f'VP resids: {all_res}')

    zk = z0_init
    q = VP.extract_sol(q_param)
    PGD_resids = []

    # print('z0_init:', z0_init)
    for k in range(K):
        # u = np.linalg.solve(lhs, zk - q)
        u = spa.linalg.spsolve(lhs, zk - q)
        utilde = proj(2 * u - zk, n)
        znew = zk + utilde - u
        # print('test zk:', znew)
        PGD_resids.append(np.max(np.abs(znew - zk)))
        zk = znew

    print('PGD resids with last K:', PGD_resids)


def proj(v, n):
    out = v.copy()
    out[n:] = np.maximum(out[n:], 0)
    return out

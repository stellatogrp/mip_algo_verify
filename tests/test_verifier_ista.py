import cvxpy as cp
import numpy as np

# from mipalgover.vector import Vector
from mipalgover.verifier import Verifier

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_ista():
    m, n = 4, 2

    np.random.seed(0)

    A = np.random.normal(size=(m, n))
    b = np.random.normal(size=(m,))
    lambd = 1

    print('testing regularization')
    x = cp.Variable(n)
    obj = .5 * cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1)
    problem = cp.Problem(cp.Minimize(obj))
    res = problem.solve()
    print(res)
    print(x.value)

    t = 0.1

    print('testing ista')
    xk = np.ones(n)
    I = np.eye(n)
    K = 1000
    At = I - t * A.T @ A
    Bt = t * A.T

    for _ in range(K):
        xk = soft_threshold(At @ xk + Bt @ b, t * lambd)

    print(xk)
    assert np.linalg.norm(x.value - xk) <= 1e-7

    VP = Verifier()
    b_offset = 0.5
    b_param = VP.add_param(m, lb=b-b_offset, ub=b+b_offset)

    z0 = VP.add_initial_iterate(n, lb=-1, ub=-1)

    K = 5
    z = [None for _ in range(K + 1)]

    z[0] = z0

    lambda_t = lambd * t
    all_res = []
    for k in range(1, K+1):
        print(k)

        z[k] = VP.soft_threshold_step(At @ z[k-1] + Bt @ b_param, lambda_t)

        VP.set_infinity_norm_objective(z[k] - z[k-1])
        res = VP.solve()
        all_res.append(res)

    print(f'opt b_param at last K: {VP.extract_sol(b_param)}')

    print(f'VP resids: {all_res}')

    zk = -np.ones(n)
    b = VP.extract_sol(b_param)
    ISTA_resids = []

    for k in range(K):
        znew = soft_threshold(At @ zk + Bt @ b, t * lambd)
        ISTA_resids.append(np.max(np.abs(znew - zk)))
        zk = znew

    print(f'ISTA resids: {ISTA_resids}')

    assert np.all(np.array(ISTA_resids) <= np.array(all_res) + 1e-7)
    assert np.abs(ISTA_resids[-1] - all_res[-1]) <= 1e-8


def soft_threshold(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

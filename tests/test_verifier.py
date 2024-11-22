import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from mipalgover.linexpr import Vector


def test_verifier():
    m, n = 2, 3
    np.random.seed(0)

    x_test = np.random.uniform(size=(n,))
    A = np.random.normal(size=(m, n))
    c = np.random.uniform(size=(n,))
    b = A @ x_test

    print('testing feasibility of LP')
    x = cp.Variable(n)
    obj = c.T @ x
    constraints = [A @ x == b, x >= 0]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    res = problem.solve()
    print(f'x_test: {x_test}')
    print(f'optimal obj: {res}')
    print(f'optimal x: {x.value}')

    print('testing with DR')

    xk = np.zeros(n)
    yk = np.zeros(m)

    def proj(v):
        return np.maximum(v, 0)

    t = 0.1
    K = 10000
    for _ in range(K):
        xnew = proj(xk - t * (c - A.T @ yk))
        ynew = yk - t * (A @ (2 * xnew - xk) - b)

        xk = xnew
        yk = ynew

    print(f'x from DR: {xk}')

    x = Vector(n)
    y = Vector(m)

    c_param = Vector(n)
    # b_param = Vector(n)

    test_expr = x - t * (c_param - A.T @ y)
    # print(test_expr.decomposition_dict)

    assert spa.linalg.norm(test_expr.decomposition_dict[x] - spa.eye(n)) <= 1e-10
    assert np.linalg.norm(test_expr.decomposition_dict[y] - t * A.T) <= 1e-10
    assert spa.linalg.norm(test_expr.decomposition_dict[c_param] + t * spa.eye(n)) <= 1e-10

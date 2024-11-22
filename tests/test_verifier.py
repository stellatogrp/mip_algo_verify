import cvxpy as cp
import numpy as np

# from mipalgover.vector import Vector
from mipalgover.verifier import Verifier


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

    VP = Verifier()

    # c_param = Verifier.add_param(n, lb=c, ub=c)
    c_param = VP.add_param(n, lb=c, ub=c)
    # b_param = VP.add_param(m, lb=b, ub=b)  # add boxes here when testing

    x0 = VP.add_initial_iterate(n, lb=0, ub=0)
    y0 = VP.add_initial_iterate(m, lb=0, ub=0)

    K = 1

    x = [None for _ in range(K+1)]
    y = [None for _ in range(K+1)]

    x[0] = x0
    y[0] = y0
    for k in range(1, K+1):
        # xk = relu(xkminus1 - t * (c_param - A.T @ yminus1))
        x[k] = VP.add_explicit_linear_step(x[k-1] - t * (c_param - A.T @ y[k-1]))

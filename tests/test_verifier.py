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
    DR_resids = []
    for _ in range(K):
        xnew = proj(xk - t * (c - A.T @ yk))
        ynew = yk - t * (A @ (2 * xnew - xk) - b)

        z_resid = np.hstack([xnew - xk, ynew - yk])
        DR_resids.append(np.max(np.abs(z_resid)))
        # DR_resids.append(np.max(np.abs(ynew - yk)))

        xk = xnew
        yk = ynew

    print(f'x from DR: {xk}')

    VP = Verifier()

    # c_param = Verifier.add_param(n, lb=c, ub=c)
    b_offset = 0
    c_param = VP.add_param(n, lb=c, ub=c)
    b_param = VP.add_param(m, lb=b-b_offset, ub=b+b_offset)  # add boxes here when testing

    # VP.add_constraint(ones @ c_param == 1)

    x0 = VP.add_initial_iterate(n, lb=0, ub=0)
    y0 = VP.add_initial_iterate(m, lb=0, ub=0)

    K = 5

    x = [None for _ in range(K+1)]
    y = [None for _ in range(K+1)]

    x[0] = x0
    y[0] = y0

    all_res = []
    for k in range(1, K+1):
        print(k)

        # x[k] = VP.add_relu_step(x[k-1] - t * (c_param - A.T @ y[k-1]))  # TODO: replace with relu once implemented
        # y[k] = VP.add_explicit_affine_step(y[k-1] - t * (A @ (2 * x[k] - x[k-1]) - b_param))

        x[k] = VP.relu_step(x[k-1] - t * (c_param - A.T @ y[k-1]))
        y[k] = y[k-1] - t * (A @ (2 * x[k] - x[k-1]) - b_param)

        VP.set_zero_objective()
        # VP.set_infinity_norm_objective(y[k] - y[k-1])
        # VP.set_infinity_norm_objective([x[k] - x[k-1], y[k] - y[k-1]])
        res = VP.solve()
        all_res.append(res)

    # print(x[1].decomposition_dict)
    # print(y[1].decomposition_dict)
    # VP.addobjective(intfy_norm(s[k] - s[k-1]))

    print(f'VP resids: {all_res}')
    print(f'DR resids: {DR_resids[:K]}')

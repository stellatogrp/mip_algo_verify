import cvxpy as cp
import numpy as np
import scipy.sparse as spa

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_pdhg_convergence():
    np.random.seed(1)
    m = 5
    n = 10
    K = 10000
    t = .1

    A = np.random.normal(size=(m, n))
    c = np.random.uniform(size=n)
    b = np.random.uniform(size=m)

    x = cp.Variable(n)

    constraints = [A @ x == b, x >= 0]
    prob = cp.Problem(cp.Minimize(c @ x), constraints)
    res = prob.solve()
    # print('--testing with cvxpy--')
    print('obj:', res)
    # print('x value:', x.value)
    # print('y value:', constraints[0].dual_value)
    cp_x = x.value
    cp_y = constraints[0].dual_value

    xk = np.zeros(n)
    yk = np.zeros(m)

    print('--testing with vanilla pdhg--')
    for _ in range(K):
        xkplus1 = np.maximum(xk - t * (c - A.T @ yk), 0)
        ykplus1 = yk - t * (A @ (2 * xkplus1 - xk) - b)

        # print(jnp.linalg.norm(ykplus1 - yk, 1) + jnp.linalg.norm(xkplus1 - xk, 1))

        xk = xkplus1
        yk = ykplus1

    print(cp_x)
    print(xk)
    print(np.linalg.norm(xk - cp_x))

    assert np.linalg.norm(xk - cp_x) <= 1e-2  # need to be pretty loose here

    assert np.linalg.norm(yk - cp_y) <= 1e-2 or np.linalg.norm(yk + cp_y) <= 1e-2 # dual var can be flipped from alg

    print('----testing matrix formulation for vanilla----')
    xk = np.zeros(n)
    yk = np.zeros(m)
    K = 10

    uk = np.zeros(n)
    vk = np.zeros(m)

    xC = spa.eye(n)
    xD = t * A.T
    xE = - t * spa.eye(n)

    vC = spa.eye(m)
    vD = -2 * t * A
    vE = t * A
    vF = t * spa.eye(m)

    for _ in range(K):
        xkplus1 = np.maximum(xk - t * (c - A.T @ yk), 0)
        ykplus1 = yk - t * (A @ (2 * xkplus1 - xk) - b)

        ukplus1 = np.maximum(xC @ uk + xD @ vk + xE @ c, 0)
        vkplus1 = vC @ vk + vD @ ukplus1 + vE @ uk + vF @ b
        # print('x:', xkplus1)
        # print('u:', ukplus1)
        assert np.linalg.norm(xkplus1 - ukplus1) <= 1e-8
        assert np.linalg.norm(ykplus1 - vkplus1) <= 1e-8

        xk = xkplus1
        yk = ykplus1
        uk = ukplus1
        vk = vkplus1

    # testing momentum
    xk = np.zeros(n)
    yk = np.zeros(m)

    uk = np.zeros(n)
    vk = np.zeros(m)
    K = 10000

    def beta_func(k):
        return k / (k + 3)
    # vk = xk
    # specific momentum from https://arxiv.org/pdf/2403.11139 with Nesterov weights

    print('testing matrix formulation for momentum')
    for k in range(K):
        xkplus1 = np.maximum(xk - t * (c - A.T @ yk), 0)
        ytilde_kplus1 = xkplus1 + k / (k + 3) * (xkplus1 - xk)
        ykplus1 = yk - t * (A @ (2 * ytilde_kplus1 - xk) - b)

        # vD = -2 * t * A
        # vE = t * A
        beta_k = beta_func(k)
        vD_k = - 2 * t * (1 + beta_k) * A
        vE_k = t * (1 + 2 * beta_k) * A

        ukplus1 = np.maximum(xC @ uk + xD @ vk + xE @ c, 0)
        vkplus1 = vC @ vk + vD_k @ ukplus1 + vE_k @ uk + vF @ b

        assert np.linalg.norm(xkplus1 - ukplus1) <= 1e-8
        assert np.linalg.norm(ykplus1 - vkplus1) <= 1e-8

        xk = xkplus1
        # vk = vkplus1
        yk = ykplus1
        uk = ukplus1
        vk = vkplus1

    assert np.linalg.norm(xk - cp_x) <= 1e-2  # need to be pretty loose here

    assert np.linalg.norm(yk - cp_y) <= 1e-2 or np.linalg.norm(yk + cp_y) <= 1e-2 # dual var can be flipped from alg

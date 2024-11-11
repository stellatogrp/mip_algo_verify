import cvxpy as cp
import numpy as np

# import matplotlib.pyplot as plt

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_portfolio():
    n = 10
    d = 3
    gamma = 3
    lambd = 1e-4

    np.random.seed(3)
    F = np.random.normal(size=(n, d))
    Fmask = np.random.randint(0, high=2, size=(n, d))
    print(Fmask)
    F = np.multiply(F, Fmask)
    Ddiag = np.random.uniform(high=np.sqrt(d), size=(n, ))
    D = np.diag(Ddiag)

    mu = np.random.normal(size=(n,))
    Sigma = F @ F.T + D
    x_prev = 1/n * np.ones(n)

    # original
    x = cp.Variable(n)
    obj = mu.T @ x - gamma * cp.quad_form(x, Sigma) - lambd * cp.sum_squares(x - x_prev)
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve()
    x_orig = x.value

    print(x_orig)

    # reformed
    x = cp.Variable(n)
    y = cp.Variable(d)
    obj = cp.quad_form(x, gamma * D + lambd * np.eye(n)) + gamma * cp.sum_squares(y) - (mu + 2 * lambd * x_prev) @ x
    constraints = [cp.sum(x) == 1, x >= 0, y == F.T @ x]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    x_reformed = x.value
    print(x_reformed)

    # block reformed
    z = cp.Variable(n + d)
    P = 2 * np.block([
        [gamma * D + lambd * np.eye(n), np.zeros((n, d))],
        [np.zeros((d, n)), gamma * np.eye(d)]
    ])
    q = np.zeros(n + d)
    q[:n] = -(mu + 2 * lambd * x_prev)

    obj = .5 * cp.quad_form(z, P) + q.T @ z
    A1 = np.block([
        [F.T, -np.eye(d)]
    ])
    b1 = np.zeros(d)

    A2 = np.block([
        [np.ones((1, n)), np.zeros((1, d))]
    ])
    b2 = 1

    A3 = np.block([
        [-np.eye(n), np.zeros((n, d))]
    ])
    b3 = np.zeros(n)

    constraints = [
        A1 @ z == b1,
        A2 @ z == b2,
        A3 @ z <= b3,
    ]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    # print(z.value)
    z_triple_block = z.value[:n]
    print(z_triple_block)

    A = np.block([
        [F.T, -np.eye(d)],
        [np.ones(n), np.zeros(d)],
        [-np.eye(n), np.zeros((n, d))]
    ])
    b = np.hstack([np.zeros(d), 1, np.zeros(n)])
    print(A)
    print(b)

    # assert A.shape == (n + d + 1, n + d)

    s = cp.Variable(n + d + 1)
    constraints = [
        A @ z + s == b,
        # s >= 0,
        s[d+1:] >= 0,
        s[:d+1] == 0,
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)
    # print(z.value)
    # print(s.value)

    z_single_block = z.value[:n]
    z_full = z.value
    y_full = constraints[0].dual_value
    print(z_single_block)
    print('full primal/dual vars:')
    print(z_full)
    print(y_full)
    print('w:', b - A @ z_full)
    print('dual feas:', P @ z_full + A.T @ y_full + q)

    assert np.linalg.norm(x_orig - x_reformed) <= 1e-7
    assert np.linalg.norm(x_reformed - z_triple_block) <= 1e-7
    assert np.linalg.norm(z_triple_block - z_single_block) <= 1e-7

    # test with DR splitting
    Am, An = A.shape
    M = np.block([
        [P, A.T],
        [-A, np.zeros((Am, Am))]
    ])
    lhs = np.eye(Am + An) + M
    c = np.hstack([q, b])
    # sk = np.zeros(Am + An)
    sk = np.hstack([z_full, y_full])

    def proj(v):
        l = Am + An - n
        u = Am + An
        v[l:u] = np.maximum(v[l:u], 0)
        return v

    print('---testing DR---')
    K = 1000
    print(c)
    for _ in range(K):
        # print('-')
        # print(sk - c)
        utilde = np.linalg.solve(lhs, sk - c)

        # print(utilde)
        # print(2*utilde - sk)
        u = proj(2 * utilde - sk)
        # print(u)
        sk = sk + u - utilde
    print('from DR:')
    # print(sk)
    z_DR = sk[:An]
    y_DR = sk[An:]
    print(z_DR)
    print(y_DR)
    print('w:', b - A @ z_DR)
    print('dual feas:', P @ z_DR + A.T @ y_full + q)


# def test_portfolio_nonnegDR():
#     n = 3
#     d = 2
#     gamma = 3
#     lambd = 1e-4

#     np.random.seed(3)
#     F = np.random.normal(size=(n, d))
#     Fmask = np.random.randint(0, high=2, size=(n, d))
#     print(Fmask)
#     F = np.multiply(F, Fmask)
#     Ddiag = np.random.uniform(high=np.sqrt(d), size=(n, ))
#     D = np.diag(Ddiag)

#     mu = np.random.normal(size=(n,))
#     Sigma = F @ F.T + D
#     x_prev = 1/n * np.ones(n)

#     # reformed
#     x = cp.Variable(n)
#     y = cp.Variable(d)
#     obj = cp.quad_form(x, gamma * D + lambd * np.eye(n)) + gamma * cp.sum_squares(y) - (mu + 2 * lambd * x_prev) @ x
#     constraints = [cp.sum(x) == 1, x >= 0, y == F.T @ x]
#     prob = cp.Problem(cp.Minimize(obj), constraints)
#     prob.solve()
#     x_reformed = x.value
#     print(x_reformed)

#     # block reformed
#     z = cp.Variable(n + d)
#     P = 2 * np.block([
#         [gamma * D + lambd * np.eye(n), np.zeros((n, d))],
#         [np.zeros((d, n)), gamma * np.eye(d)]
#     ])
#     q = np.zeros(n + d)
#     q[:n] = -(mu + 2 * lambd * x_prev)

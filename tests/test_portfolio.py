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

    assert np.linalg.norm(x_orig - x_reformed) <= 1e-5
    assert np.linalg.norm(x_reformed - z_triple_block) <= 1e-5
    assert np.linalg.norm(z_triple_block - z_single_block) <= 1e-5

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
        # l = Am + An - n
        l = An + d + 1
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


def test_portfolio_l1():
    print('----testing l1----')
    n = 10
    d = 3
    gamma = 5
    alpha = 1 / gamma
    lambd = 1

    np.random.seed(3)
    F = np.random.normal(size=(n, d))
    Fmask = np.random.randint(0, high=2, size=(n, d))
    F = np.multiply(F, Fmask)
    Ddiag = np.random.uniform(high=np.sqrt(d), size=(n, ))
    D = np.diag(Ddiag)

    mu = np.random.normal(size=(n,))
    Sigma = F @ F.T + D
    x_prev = 1/n * np.ones(n)

    # original
    C = 1.5 / n
    x = cp.Variable(n)
    obj = mu.T @ x - gamma * cp.quad_form(x, Sigma) - lambd * cp.norm(x - x_prev, 1)
    constraints = [cp.sum(x) == 1, x >= 0, x <= C]
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve()
    x_orig = x.value

    print(x_orig)

    # reformed
    z = cp.Variable(2 * n + d)

    A = np.block([
        [F.T, -np.eye(d), np.zeros((d, n))],
        [np.ones((1, n)), np.zeros((1, d)), np.zeros((1, n))],
        [np.eye(n), np.zeros((n, d)), -np.eye(n)],
        [-np.eye(n), np.zeros((n, d)), -np.eye(n)],
        [-np.eye(n), np.zeros((n, d)), np.zeros((n, n))],
        [np.eye(n), np.zeros((n, d)), np.zeros((n, n))],
    ])

    print(A.shape)
    print(4 * n + d + 1)
    print(2 * n + d)

    b = np.hstack([np.zeros(d), 1, x_prev, -x_prev, np.zeros(n), C * np.ones(n)])

    # s = cp.Variable(3 * n + d + 1)
    s = cp.Variable(4 * n + d + 1)

    q = np.hstack([-alpha * mu, np.zeros(d), alpha * lambd * np.ones(n)])
    print(q.shape)

    # P = 2 * np.block([
    #     [gamma * D + lambd * np.eye(n), np.zeros((n, d))],
    #     [np.zeros((d, n)), gamma * np.eye(d)]
    # ])

    P = 2 * np.block([
        [D, np.zeros((n, d)), np.zeros((n, n))],
        [np.zeros((d, n)), np.eye(d), np.zeros((d, n))],
        [np.zeros((n, n)), np.zeros((n, d)), np.zeros((n, n))]
    ])

    obj = .5 * cp.quad_form(z, P) + q @ z
    constraints = [
        A @ z + s == b,
        s[:d+1] == 0,
        s[d+1:] >= 0,
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    print(z.value)

    # z_full = z.value
    # y_full = constraints[0].dual_value

    assert np.linalg.norm(x_orig - z.value[:n]) <= 1e-5

    print('solving with DR')

    Am, An = A.shape
    M = np.block([
        [P, A.T],
        [-A, np.zeros((Am, Am))]
    ])
    lhs = np.eye(Am + An) + M
    c = np.hstack([q, b])
    sk = np.zeros(Am + An)
    # sk = np.hstack([z.value, y_full])

    def proj(v):
        # l = Am + An - (3 * n)
        l = An + d + 1
        u = Am + An
        v[l:u] = np.maximum(v[l:u], 0)
        return v

    K = 1000
    for _ in range(K):
        # print('-')
        # print(sk - c)
        utilde = np.linalg.solve(lhs, sk - c)

        # print(utilde)
        # print(2*utilde - sk)
        u = proj(2 * utilde - sk)
        # print(u)
        sk = sk + u - utilde

    print(sk)
    assert np.linalg.norm(sk[:n] - x_orig) <= 1e-5


def test_sampling():
    n = 10
    C = 2 / n

    N = 10000
    samples = np.random.dirichlet(np.ones(n), size=N)
    print(samples)
    filtered_samples = samples[np.all(samples <= C, axis=1)]
    # mask = samples[]
    print(filtered_samples.shape)
    # print(filtered_samples)

    assert np.all(filtered_samples <= C)

import cvxpy as cp
import numpy as np

# import matplotlib.pyplot as plt

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_portfolio():
    n = 3
    d = 2
    gamma = 3
    lambd = 1e-4

    np.random.seed(3)
    F = np.random.normal(size=(n, d))
    Fmask = np.random.randint(0, high=2, size=(n, d))
    print(Fmask)
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

    A = np.block([
        [F.T, -np.eye(d)],
        [np.ones((1, n)), np.zeros((1, d))],
        [-np.eye(n), np.zeros((n, d))]
    ])
    b = np.hstack([np.zeros(d), 1, np.zeros(n)])
    print(A)
    print(b)

    # assert A.shape == (n + d + 1, n + d)

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

    s = cp.Variable(n + d + 1)
    constraints = [
        A @ z + s == b,
        s >= 0,
        s[:d+1] == 0,
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    # print(z.value)
    # print(s.value)

    z_single_block = z.value[:n]
    print(z_single_block)

    assert np.linalg.norm(x_orig - x_reformed) <= 1e-7
    assert np.linalg.norm(x_reformed - z_triple_block) <= 1e-7
    assert np.linalg.norm(z_triple_block - z_single_block) <= 1e-7

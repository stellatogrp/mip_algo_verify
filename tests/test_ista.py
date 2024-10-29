import cvxpy as cp
import numpy as np

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_convergence():
    m = 10
    n = 15
    K = 10000

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    b = np.random.uniform(size=(m,))

    ATA = A.T @ A
    L = np.max(np.linalg.eigvals(ATA))
    print(L)

    t = 1 / L

    def soft_threshold(x, gamma):
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    lambd = 1e-2

    x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    print(x_lstsq)

    x = cp.Variable(n)
    obj = cp.Minimize(.5 * cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1))
    prob = cp.Problem(obj)
    prob.solve()
    print(x.value)

    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T

    xk = np.zeros(n)
    lambda_t = lambd * t

    for _ in range(K):
        xk = soft_threshold(At @ xk + Bt @ b, lambda_t)

    print(xk)
    assert np.linalg.norm(xk - x.value) <= 1e-7

    print('--testing fista--')

    betak = 1.
    xk = np.zeros(n)
    wk = xk
    for _ in range(K):
        xnew = soft_threshold(At @ wk + Bt @ b, lambda_t)
        beta_new = .5 * (1 + np.sqrt(1 + 4 * betak ** 2))
        wnew = xnew + (betak - 1)/beta_new * (xnew - xk)

        xk = xnew
        betak = beta_new
        wk = wnew

    print(xk)
    assert np.linalg.norm(xk - x.value) <= 1e-7

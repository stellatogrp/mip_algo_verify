import cvxpy as cp
import networkx as nx
import numpy as np
import scipy.sparse as spa

# import matplotlib.pyplot as plt

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_flow():
    n_supply = 4
    n_demand = 2
    p = 0.6
    seed = 1

    G = nx.bipartite.random_graph(n_supply, n_demand, p, seed=seed, directed=False)

    A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)

    A[n_supply:, :] *= -1
    print(A.todense())

    n_nodes, n_arcs = A.shape
    print(f'num_arcs: {n_arcs}')

    supply_max = 10
    # demand_lb = -5
    demand_ub = -4
    capacity = 4

    np.random.seed(seed)
    c = np.random.uniform(low=1, high=10, size=n_arcs)
    A_supply = A[:n_supply, :]
    A_demand = A[n_supply:, :]
    b_supply = supply_max * np.ones(n_supply)
    b_demand = demand_ub * np.ones(n_demand)
    u = capacity * np.ones(n_arcs)

    print(A_supply.todense())
    print(A_demand.todense())

    x = cp.Variable(n_arcs)
    obj = cp.Minimize(c.T @ x)
    constraints = [
        A_supply @ x <= b_supply,
        A_demand @ x == b_demand,
        x >= 0,
        x <= u,
    ]
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.CLARABEL)

    print(res)
    print(x.value)
    cp_x = x.value

    A_block = spa.bmat([
        [A_supply, spa.eye(n_supply), None],
        [A_demand, None, None],
        [spa.eye(n_arcs), None, spa.eye(n_arcs)]
    ])

    # print(A_block.todense())
    print(A_block.shape)

    print(n_supply + n_demand + n_arcs)
    print(n_supply + 2 * n_arcs)

    assert A_block.shape == (n_supply + n_demand + n_arcs, n_supply + 2 * n_arcs)

    n_tilde = A_block.shape[1]
    x_tilde = cp.Variable(n_tilde)
    c_tilde = np.zeros(n_tilde)
    c_tilde[:n_arcs] = c

    print(c_tilde)
    b_tilde = np.hstack([b_supply, b_demand, u])

    obj = cp.Minimize(c_tilde.T @ x_tilde)
    constraints = [A_block @ x_tilde == b_tilde, x_tilde >= 0]

    prob = cp.Problem(obj, constraints)
    res2 = prob.solve(solver=cp.CLARABEL)
    print(res2)
    print(x_tilde.value)

    assert np.abs(res - res2) <= 1e-6

    print('2-norm of A:', spa.linalg.norm(A_block, ord=2))

    t = 0.4 / spa.linalg.norm(A_block, ord=2)
    print('t:', t)

    m, n = A_block.shape

    xk = np.zeros(n)
    yk = np.zeros(m)
    K = 10000

    print('--testing with vanilla pdhg--')
    for _ in range(K):
        xkplus1 = np.maximum(xk - t * (c_tilde - A_block.T @ yk), 0)
        ykplus1 = yk - t * (A_block @ (2 * xkplus1 - xk) - b_tilde)

        # print(jnp.linalg.norm(ykplus1 - yk, 1) + jnp.linalg.norm(xkplus1 - xk, 1))

        xk = xkplus1
        yk = ykplus1

    print(xk)
    print('norm diff:', np.linalg.norm(xk - x_tilde.value))
    assert np.linalg.norm(xk - x_tilde.value) <= 1e-6

    def beta_func(k):
        return k / (k+3)

    def get_vDk_vEk(k, A, t, momentum=True):
        vD = -2 * t * A
        vE = t * A

        if momentum:
            beta_k = beta_func(k)
            vD_k = -2 * t * (1 + beta_k) * A
            vE_k = t * (1 + 2 * beta_k) * A
        else:
            vD_k = vD
            vE_k = vE
        return vD_k, vE_k
    print('--testing with momentum pdhg--')

    xC = spa.eye(n)
    xD = t * A_block.T
    xE = - t * spa.eye(n)

    vC = spa.eye(m)
    # vD = -2 * t * A
    # vE = t * A
    vF = t * spa.eye(m)

    uk = np.zeros(n)
    vk = np.zeros(m)

    for k in range(K):

        vD_k, vE_k = get_vDk_vEk(k, A_block, t)

        ukplus1 = np.maximum(xC @ uk + xD @ vk + xE @ c_tilde, 0)
        vkplus1 = vC @ vk + vD_k @ ukplus1 + vE_k @ uk + vF @ b_tilde

        # xk = xkplus1
        # vk = vkplus1
        # yk = ykplus1
        uk = ukplus1
        vk = vkplus1
    print(ukplus1)

    assert np.linalg.norm(uk - x_tilde.value) <= 1e-6

    satlin_pdhg(c, A_supply, A_demand, b_supply, b_demand, u, cp_x)
    satlin_pdhg_momentum(c, A_supply, A_demand, b_supply, b_demand, u, cp_x)
    satlin_split_pdhg(c, A_supply, A_demand, b_supply, b_demand, u, cp_x)
    satlin_split_pdhg_momentum(c, A_supply, A_demand, b_supply, b_demand, u, cp_x)


def satlin_pdhg(c, A_supply, A_demand, b_supply, b_demand, u, cp_x):
    print('cp_x:', cp_x)
    K = spa.vstack([-A_supply, A_demand])
    q = np.hstack([-b_supply, b_demand])
    t = 0.4 / spa.linalg.norm(K, ord=2)
    print('t:', t)
    print('c:', c)
    print('u:', u)

    zk = np.zeros(K.shape[1])
    yk = np.zeros(K.shape[0])

    K_max = 10000
    for _ in range(K_max):
        zkplus1 = satlin(zk - t * (c - K.T @ yk), 0, u)
        ykplus1 = proj(yk + t * (q - K @ (2 * zkplus1 - zk)), A_supply.shape[0])

        zk = zkplus1
        yk = ykplus1

    print('zK:', zk)

    assert np.linalg.norm(cp_x - zk) <= 1e-6


def satlin_pdhg_momentum(c, A_supply, A_demand, b_supply, b_demand, u, cp_x):
    print('momentum test')
    K = spa.vstack([-A_supply, A_demand])
    q = np.hstack([-b_supply, b_demand])
    t = 0.4 / spa.linalg.norm(K, ord=2)
    print('t:', t)
    print('c:', c)
    print('u:', u)

    zk = np.zeros(K.shape[1])
    # ztilde_k = np.zeros(K.shape[1])
    yk = np.zeros(K.shape[0])

    K_max = 10000
    for k in range(1, K_max+1):
        zkplus1 = satlin(zk - t * (c - K.T @ yk), 0, u)
        ztilde_kplus1 = zkplus1 + k / (k + 3) * (zkplus1 - zk)
        ykplus1 = proj(yk + t * (q - K @ (2 * ztilde_kplus1 - zk)), A_supply.shape[0])

        zk = zkplus1
        yk = ykplus1

    print('zK:', zk)
    assert np.linalg.norm(cp_x - zk) <= 1e-6


def satlin_split_pdhg(c, A_supply, A_demand, b_supply, b_demand, u, cp_x):
    print('vanilla satlin with split dual vars')
    # neg_A_supply = -A_supply
    # neg_b_supply = -b_supply
    K = spa.vstack([-A_supply, A_demand])

    t = 0.4 / spa.linalg.norm(K, ord=2)

    zk = np.zeros(K.shape[1])
    yk_1 = np.zeros(A_supply.shape[0])
    yk_2 = np.zeros(A_demand.shape[0])

    K_max = 10000
    for k in range(1, K_max + 1):
        zkplus1 = satlin(zk - t * (c + A_supply.T @ yk_1 - A_demand.T @ yk_2), 0, u)
        ykplus1_1 = np.maximum(yk_1 + t * (-b_supply + A_supply @ (2 * zkplus1 - zk)), 0)
        ykplus1_2 = yk_2 + t * (b_demand - A_demand @ (2 * zkplus1 - zk))

        zk = zkplus1
        yk_1 = ykplus1_1
        yk_2 = ykplus1_2
    print('zK:', zk)
    assert np.linalg.norm(cp_x - zk) <= 1e-6


def satlin_split_pdhg_momentum(c, A_supply, A_demand, b_supply, b_demand, u, cp_x):
    print('vanilla satlin with split dual vars with momentum')
    K = spa.vstack([-A_supply, A_demand])

    t = 0.4 / spa.linalg.norm(K, ord=2)

    zk = np.zeros(K.shape[1])
    yk_1 = np.zeros(A_supply.shape[0])
    yk_2 = np.zeros(A_demand.shape[0])

    K_max = 10000
    for k in range(1, K_max + 1):
        zkplus1 = satlin(zk - t * (c + A_supply.T @ yk_1 - A_demand.T @ yk_2), 0, u)
        ztilde_kplus1 = zkplus1 + k / (k + 3) * (zkplus1 - zk)
        ykplus1_1 = np.maximum(yk_1 + t * (-b_supply + A_supply @ (2 * ztilde_kplus1 - zk)), 0)
        ykplus1_2 = yk_2 + t * (b_demand - A_demand @ (2 * ztilde_kplus1 - zk))

        zk = zkplus1
        yk_1 = ykplus1_1
        yk_2 = ykplus1_2
    print('zK:', zk)
    assert np.linalg.norm(cp_x - zk) <= 1e-6


def proj(v, m1):
    out = v.copy()
    out[:m1] = np.maximum(out[:m1], 0)
    return out


def satlin(v, l, u):
    return np.minimum(np.maximum(v, l), u)

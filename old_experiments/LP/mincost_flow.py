import cvxpy as cp
import networkx as nx
import numpy as np
import scipy.sparse as spa

# import matplotlib.pyplot as plt

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def main():
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
    res = prob.solve()

    print(res)
    print(x.value)

    A_block = spa.bmat([
        [A_supply, spa.eye(n_supply), None],
        [A_demand, None, None],
        [spa.eye(n_arcs), None, spa.eye(n_arcs)]
    ])

    # print(A_block.todense())
    print(A_block.shape)

    print(n_supply + n_demand + n_arcs)
    print(n_supply + 2 * n_arcs)

    n_tilde = A_block.shape[1]
    x_tilde = cp.Variable(n_tilde)
    c_tilde = np.zeros(n_tilde)
    c_tilde[:n_arcs] = c

    print(c_tilde)
    b_tilde = np.hstack([b_supply, b_demand, u])

    obj = cp.Minimize(c_tilde.T @ x_tilde)
    constraints = [A_block @ x_tilde == b_tilde, x_tilde >= 0]

    prob = cp.Problem(obj, constraints)
    res2 = prob.solve()
    print(res2)
    print(x_tilde.value)

    print('2-norm of A:', spa.linalg.norm(A_block, ord=2))

    t = 0.9 / spa.linalg.norm(A_block, ord=2)
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


if __name__ == '__main__':
    main()

import cvxpy as cp
import networkx as nx
import numpy as np

# import matplotlib.pyplot as plt

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def main():
    n_supply = 4
    n_demand = 4
    p = 0.6
    seed = 1

    G = nx.bipartite.random_graph(n_supply, n_demand, p, seed=seed, directed=False)

    A = nx.linalg.graphmatrix.incidence_matrix(G, oriented=False)
    A[n_supply:, :] *= -1
    print(A.todense())

    n_nodes, n_arcs = A.shape

    supply_max = 10
    # demand_lb = -5
    demand_ub = -3
    capacity = 5

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


if __name__ == '__main__':
    main()

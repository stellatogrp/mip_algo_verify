
import gurobipy as gp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def test_simple_l1(n, l, u):
    model = gp.Model()

    x, up, un, v = {}, {}, {}, {}

    for i in range(n):
        Ui = u[i]
        Li = l[i]
        Mi = jnp.max(jnp.abs(jnp.array([Ui, Li])))

        print(Li, Ui)

        x[i] = model.addVar(lb=l[i], ub=u[i])

        up[i] = model.addVar(lb=0, ub=Mi)
        un[i] = model.addVar(lb=0, ub=Mi)
        v[i] = model.addVar(vtype=gp.GRB.BINARY)
        model.addConstr(up[i] - un[i] == x[i])
        model.addConstr(up[i] <= Ui*v[i])
        model.addConstr(un[i] <= jnp.abs(Li)*(1 - v[i]))

    model.setObjective(gp.quicksum((up[i] + un[i]) for i in range(n)), gp.GRB.MAXIMIZE)
    model.optimize()
    print(up[0].X, un[0].X, up[1].X, un[1].X)


# def test_l1_diff(n, x_l, x_u, y_l, y_u):
#     up, un, v = {}, {}, {}

#     for i in range(n):
#         Ui = y_u[i] - x_l[i]
#         Li = y_l[i] - x_u[i]
#         print(Li, Ui)


def main():
    n = 2

    l = jnp.array([-2, 1])
    u = jnp.array([1, 3])

    test_simple_l1(n, l, u)

    # x_l = jnp.array([-2, -2])
    # x_u = jnp.array([-1, 1])

    # y_l = jnp.array([-2, -1])
    # y_u = jnp.array([1, 1])

    # test_l1_diff(n, x_l, x_u, y_l, y_u)


if __name__ == '__main__':
    main()

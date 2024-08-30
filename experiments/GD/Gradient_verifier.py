import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB, Model, max_, quicksum


def Gradient_verifier(cz, k, pnorm):
    n = 2

    # Fixed parameters
    mu = 1
    L = 10
    r = 0.1
    # Data
    I = np.identity(2)
    t = 2/(mu + L)
    P = np.matrix([[mu, 0], [0, L]])
    ItP = I - t*P
    H = np.linalg.matrix_power(ItP, k) - np.linalg.matrix_power(ItP, k-1)

    # Create a new model
    model = Model()
    model.Params.NumericFocus = 3

    # Create variables
    z = {}
    up, un, v = {}, {}, {}
    pp, pn, q = {}, {}, {}

    for i in range(n):
        z[i] = model.addVar(lb=-1, ub=1, name="z%d" % i)
        # abs value for objective
        v[i] = model.addVar(vtype=GRB.BINARY, name="v%d" % i)
        up[i] = model.addVar(lb=0, name="up%d" % i)
        un[i] = model.addVar(lb=0, name="un%d" % i)
        # abs value for constraints
        q[i] = model.addVar(vtype=GRB.BINARY, name="q%d" % i)
        pp[i] = model.addVar(lb=0, name="pp%d" % i)
        pn[i] = model.addVar(lb=0, name="pn%d" % i)

    # Constraints
    M = 10000
    for i in range(n):
        model.addConstr(up[i] - un[i] == H[i, i]*z[i])
        model.addConstr(up[i] <= M*v[i])
        model.addConstr(un[i] <= M*(1 - v[i]))

    for i in range(n):
        model.addConstr(pp[i] - pn[i] == z[i] - cz[i])
        model.addConstr(pp[i] <= M*q[i])
        model.addConstr(pn[i] <= M*(1 - q[i]))

    if pnorm == 1:
        # Radius constraint on z start (l1-ball)
        model.addConstr(quicksum((pp[i]+pn[i]) for i in range(n)) <= r)
        # L1 norm objective
        model.setObjective(quicksum((up[i] + un[i]) for i in range(n)), GRB.MAXIMIZE)

    elif pnorm == 2:
        # Radius constraint on z start (l2-ball)
        model.addConstr(quicksum((pp[i]+pn[i])*(pp[i]+pn[i]) for i in range(n)) <= r*r)
        # L2 norm objective
        model.setObjective(quicksum((up[i] + un[i])*(up[i] + un[i]) for i in range(n)), GRB.MAXIMIZE)
    else:
        # Radius constraint on z start (l_inf-ball)
        Ball = model.addVar(ub=r)
        Q = [model.addVar() for i in range(n)]
        for i in range(n):
            model.addConstr(Q[i] == pp[i] + pn[i])
        model.addConstr(Ball == max_(Q))
        # model.addConstr(Ball <= r) # -> already in the upper bound of Ball

        # Linf norm objective
        U = [model.addVar() for i in range(n)]
        for i in range(n):
            model.addConstr(U[i] == up[i] + un[i])
        Obj = model.addVar()
        # TODO: replace with the formulation written in Overleaf
        model.addConstr(Obj == max_(U))
        model.setObjective(Obj, GRB.MAXIMIZE)

    model.optimize()

    return model.objVal


def Plot(R, msg=""):
    S = {1: 's', 2: 'p', 3: '+'}

    for i in R.keys():
        plt.plot(range(1, 1+len(R[i])), R[i], label=msg+"$_%d$" % i, marker=S[i])

    plt.yscale('log')
    plt.xlabel("k")
    plt.ylabel("obj")
    plt.legend()
    plt.title("Residuals norm: $||z_{k+1} - z_k||$")
    plt.show()

if __name__ == '__main__':
    K = 25
    pnorm = 3

    c1 = np.array([0,1])
    c2 = np.array([0.842, -0.317])
    c3 = np.array([-0.397, -0.807])

    R = {}
    # Change starting point
    R[1], R[2], R[3] = [], [], []
    for k in range(1, K+1):
        R[1].append(Gradient_verifier(c1, k, pnorm))
    for k in range(1, K+1):
        R[2].append(Gradient_verifier(c2, k, pnorm))
    for k in range(1, K+1):
        R[3].append(Gradient_verifier(c3, k, pnorm))

    #Plot(R, "Z")

    # Change norm, fix Z1
    R[1], R[2], R[3] = [], [], []
    for k in range(1, K+1):
        R[1].append(Gradient_verifier(c2, k, 1))
    for k in range(1, K+1):
        R[2].append(Gradient_verifier(c2, k, 2))
    for k in range(1, K+1):
        R[3].append(Gradient_verifier(c2, k, 8)) # l_inf norm

    Plot(R, "p")

import numpy as np
from gurobipy import GRB, Model, max_, quicksum


def VerifyISTA_withBounds(K, pnorm, At, Bt, lambda_t, z_0, c_theta, r_theta, Deltas, y_LB, y_UB, z_LB, z_UB, zbar, ybar, thetabar):
    n = len(z_0)
    m = len(c_theta)

    model = Model()
    #model.Params.NumericFocus = 3

    # Create variables
    z, y = {}, {}
    up, un, v = {}, {}, {}

    for k in range(K+1):
        for i in range(n):
            z[i,k] = model.addVar(lb=z_LB[i,k], ub=z_UB[i,k])
            if k > 0:
                y[i,k] = model.addVar(lb=y_LB[i,k], ub=y_UB[i,k])

    theta = {}
    for h in range(m):
        theta[h] = model.addVar(lb=c_theta[h]-r_theta, ub=c_theta[h]+r_theta)

    for i in range(n):
        # abs value for objective
        v[i] = model.addVar(vtype=GRB.BINARY)
        up[i] = model.addVar(lb=0, ub=z_UB[i, K] - z_LB[i, K-1])
        un[i] = model.addVar(lb=0, ub=z_UB[i, K-1] - z_LB[i, K])

    w1, w2 = {}, {}
    for k in range(1, K+1):
        for i in range(n):
            # Soft thresholding
            w1[i, k] = model.addVar(vtype=GRB.BINARY)
            w2[i, k] = model.addVar(vtype=GRB.BINARY)

    model.update()

    # Set objective
    if pnorm == 2:
        # Euclidean norm (p=2)
        model.Params.NonConvex = 2
        model.setObjective(quicksum((up[i] + un[i])*(up[i] + un[i]) for i in range(n)), GRB.MAXIMIZE)
    elif pnorm == 1:
        # Mannhattan norm (p=1)
        model.setObjective(quicksum((up[i] + un[i]) for i in range(n)), GRB.MAXIMIZE)
    else:
        # Infinit norm (p=inf)
        U = [model.addVar() for i in range(n)]
        for i in range(n):
            model.addConstr(U[i] == up[i] + un[i])
        Obj = model.addVar()
        # TODO: replace with the formulation written in Overleaf
        model.addConstr(Obj == max_(U))
        model.setObjective(Obj, GRB.MAXIMIZE)

    # Constraints fun obj
    for i in range(n):
        model.addConstr(up[i] - un[i] == z[i, K] - z[i, K-1])
        model.addConstr(up[i] <= (z_UB[i,K] - z_LB[i,K-1])*v[i])
        model.addConstr(un[i] <= (z_UB[i,K-1] - z_LB[i,K])*(1 - v[i]))

    # Constraints affine step
    for k in range(K):
        for i in range(n):
            model.addConstr(y[i,k+1] == quicksum(At[i,j]*z[j,k] for j in range(n)) + quicksum(Bt[i,h]*theta[h] for h in range(m)))

    # Constraints on l1-norm regularaizer
    for k in range(K):
        for i in range(n):
            # box-bound constraints (polyhedral relaxation of soft-thresholding)
            # TODO: reason about better numerical tollerances (in the following 4 lines)
            if z_UB[i,k+1] < -0.01:
                model.addConstr(z[i, k+1] == y[i, k+1] + lambda_t)
            elif z_LB[i,k+1] > 0.01:
                model.addConstr(z[i, k+1] == y[i, k+1] - lambda_t)
            elif z_UB[i,k+1] < 0.00001 and z_LB[i,k+1] > -0.00001:
                model.addConstr(z[i, k+1] == 0.0)
            else:
                model.addConstr(z[i,k+1] >= y[i,k+1] - lambda_t)
                model.addConstr(z[i,k+1] <= y[i,k+1] + lambda_t)
                model.addConstr(z[i,k+1] <= z_UB[i,k+1]/(z_UB[i,k+1] + 2*lambda_t)*(y[i,k+1] + lambda_t))
                model.addConstr(z[i,k+1] >= z_LB[i,k+1]/(z_LB[i,k+1] - 2*lambda_t)*(y[i,k+1] - lambda_t))

                # Upper right part: w1 = 1, y >= lambda_t
                model.addConstr(z[i,k+1] <= z_UB[i,k+1]*w1[i,k+1])
                model.addConstr(z[i,k+1] <= y[i,k+1] - lambda_t + 2*lambda_t*(1-w1[i,k+1]))
                model.addConstr(y[i,k+1] >= lambda_t - (2*lambda_t - z_LB[i,k+1])*(1-w1[i,k+1]))
                model.addConstr(y[i,k+1] <= lambda_t + z_UB[i,k+1]*w1[i,k+1])

                # Lower left part: w1 = 1, y <= -lambda_t
                model.addConstr(z[i,k+1] >= z_LB[i,k+1]*w2[i,k+1])
                model.addConstr(z[i,k+1] >= y[i,k+1] + lambda_t - 2*lambda_t*(1-w2[i,k+1]))
                model.addConstr(y[i,k+1] <= -lambda_t + (2*lambda_t + z_UB[i,k+1])*(1-w2[i,k+1]))
                model.addConstr(y[i,k+1] >= -lambda_t + z_LB[i,k+1]*w2[i,k+1])

                # The left and right part cannot be hold at the same time (improve LP relaxation)
                model.addConstr(w1[i,k+1] + w2[i,k+1] <= 1)

    # Complete previous solution
    if zbar is not None:
        for i, k in zbar:
            z[i, k].Start = zbar[i,k]
        for i, k in ybar:
            y[i, k].Start = ybar[i,k]
        for h in thetabar:
            theta[h].Start = thetabar[h]

    model.update()

    # Dump the model for debug (maybe with some open source solver)
    model.write('ista_{}.lp'.format(str(K)))

    # Solve the model
    model.optimize()

    # Check the status
    # TODO: add a time limit and check the status
    if model.status != GRB.OPTIMAL:
        print('model status:', model.status)
        return None

    return model.objVal, {(i,k): z[i,k].X for i, k in z}, {(i,k): y[i,k].X for i, k in y}, {j: theta[j].X for j in theta}


def BoundTightY(K, At, Bt, lambda_t, z_0, c_theta, r_theta, basic=False):
    n = len(z_0)
    m = len(c_theta)

    # First step: init lb and ub with standard techniques
    # keep the bounds into a dictionary
    y_LB, y_UB = {}, {}
    z_LB, z_UB = {}, {}
    for i in range(n):
        z_UB[i, 0] = z_0[i]
        z_LB[i, 0] = z_0[i]

    for q in range(1, K+1):
        for i in range(n):
            y_UB[i, q]  = sum(At[i, j]*z_UB[j, q-1] for j in range(n) if At[i, j] > 0)
            y_UB[i, q] += sum(At[i, j]*z_LB[j, q-1] for j in range(n) if At[i, j] < 0)
            y_UB[i, q] += sum(Bt[i, h]*(c_theta[h]+r_theta) for h in range(m) if Bt[i, h] > 0)
            y_UB[i, q] += sum(Bt[i, h]*(c_theta[h]-r_theta) for h in range(m) if Bt[i, h] < 0)

            y_LB[i, q]  = sum(At[i, j]*z_LB[j, q-1] for j in range(n) if At[i, j] > 0)
            y_LB[i, q] += sum(At[i, j]*z_UB[j, q-1] for j in range(n) if At[i, j] < 0)
            y_LB[i, q] += sum(Bt[i, h]*(c_theta[h]-r_theta) for h in range(m) if Bt[i, h] > 0)
            y_LB[i, q] += sum(Bt[i, h]*(c_theta[h]+r_theta) for h in range(m) if Bt[i, h] < 0)

            z_LB[i, q] = y_LB[i, q] - lambda_t if y_LB[i, q] < 0 else 0
            z_UB[i, q] = y_UB[i, q] + lambda_t if y_UB[i, q] > 0 else 0

    if basic:
        return y_LB, y_UB, z_LB, z_UB

    for kk in range(1, K+1):
        print('^^^^^^^^^^^^ Bound tighting, K =', kk, '^^^^^^^^^^')
        for ii in range(n):
            for sense in [GRB.MAXIMIZE, GRB.MINIMIZE]:
                model = Model()
                model.Params.OutputFlag = 0
                #model.Params.NumericFocus = 3

                # Create variables
                z, y = {}, {}
                for k in range(K+1):
                    for i in range(n):
                        # Iterates variables
                        if k == 0:
                            z[i, k] = model.addVar(lb=z_0[i], ub=z_0[i])
                        else:
                            z[i, k] = model.addVar(lb=z_LB[i, k], ub=z_UB[i, k])
                            y[i, k] = model.addVar(lb=y_LB[i, k], ub=y_UB[i, k])

                theta = {}
                for h in range(m):
                    theta[h] = model.addVar(lb=c_theta[h]-r_theta, ub=c_theta[h]+r_theta)

                model.setObjective(y[ii, kk], sense)

                # Constraints affine step
                for k in range(K):
                    for i in range(n):
                        model.addConstr(y[i, k+1] == quicksum(At[i,j]*z[j, k] for j in range(n)) + quicksum(Bt[i, h]*theta[h] for h in range(m)))
                # Constraints on l1-norm regularizer
                for k in range(K):
                    for i in range(n):
                        # box-bound constraints (polyhedral relaxation of soft-thresholding)
                        model.addConstr(z[i,k+1] >= y[i,k+1] - lambda_t)
                        model.addConstr(z[i,k+1] <= y[i,k+1] + lambda_t)

                        if z_UB[i, k+1] > 0.01:
                            model.addConstr(z[i,k+1] <= z_UB[i, k+1]/(z_UB[i, k+1] + 2*lambda_t)*(y[i,k+1] + lambda_t))
                        elif z_UB[i, k+1] < -0.01:
                            model.addConstr(z[i, k+1] == y[i, k+1] + lambda_t)

                        if z_LB[i, k+1] < -0.01:
                            model.addConstr(z[i,k+1] >= z_LB[i, k+1]/(z_LB[i, k+1] - 2*lambda_t)*(y[i,k+1] - lambda_t))
                        elif z_LB[i, k+1] > 0.01:
                            model.addConstr(z[i,k+1] == y[i,k+1] - lambda_t)

                model.optimize()

                if model.status != GRB.OPTIMAL:
                    print('bound tighting failes, GRB model status:', model.status)
                    return None

                # Update bounds
                obj = model.objVal
                if sense == GRB.MAXIMIZE:
                    y_UB[ii, kk] = min(y_UB[ii, kk], obj)
                    z_UB[ii, kk] = min(z_UB[ii, kk], y_UB[ii, kk] + lambda_t)

                    model.setAttr(GRB.Attr.UB, y[ii, kk], y_UB[ii, kk])
                    model.setAttr(GRB.Attr.UB, z[ii, kk], z_UB[ii, kk])
                else:
                    y_LB[ii, kk] = max(y_LB[ii, kk], obj)
                    z_LB[ii, kk] = max(z_LB[ii, kk], y_LB[ii, kk] - lambda_t)

                    model.setAttr(GRB.Attr.LB, y[ii, kk], y_LB[ii, kk])
                    model.setAttr(GRB.Attr.LB, z[ii, kk], z_LB[ii, kk])

                model.update()

    return y_LB, y_UB, z_LB, z_UB


def Generate_A_mat(m,n,seed):
    np.random.seed(seed)
    return np.random.randn(m, n)

def MakeData(best_t=False):
    seed = 3

    lambd = 10
    t = 0.04

    m, n = 10,15
    A = Generate_A_mat(m, n, seed)

    if best_t:
        ATA = A.T @ A
        eigs = np.linalg.eigvals(ATA)
        mu = np.min(eigs)
        L = np.max(eigs)
        t = np.real(2 / (mu + L))

    lambda_t = lambd*t

    At = np.eye(n) - t*(A.T @ A)
    Bt = t*A.T

    c_theta = 10 * np.ones((m, 1))
    r_theta = 0.25

    c_theta = c_theta[:, 0]

    c_z, _, _, _ = np.linalg.lstsq(A, c_theta + np.random.uniform(low=-r_theta, high=r_theta, size=m), rcond=None)
    c_z = c_z.reshape(-1)

    return At, Bt, lambda_t, c_z, c_theta, r_theta


if __name__ == '__main__':
    At, Bt, lambda_t, c_z, c_theta, r_theta = MakeData()

    # Number of iterations
    K = 10
    pnorm = 1

    # Basic or advanced bound tightening
    y_LB, y_UB, z_LB, z_UB = BoundTightY(K, At, Bt, lambda_t, c_z, c_theta, r_theta, basic=False)

    # Iterative
    Deltas = []
    zbar, ybar, thetabar = None, None, None
    for k in range(1, K+1):
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VerifyISTA_withBounds, K =', k, '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        delta_k, zbar, ybar, thetabar = VerifyISTA_withBounds(k, pnorm, At, Bt, lambda_t, c_z, c_theta, r_theta, Deltas, y_LB, y_UB, z_LB, z_UB, zbar, ybar, thetabar)
        Deltas.append(delta_k)
        # Dump for plotting later with pgfplots
        print('Residual for pgfplots:', Deltas)

    # TODO: add perf_couter to measure overall runtime


# TESTING: Check the output for K=10 with the following logs

# -------------------------------------------------------------
# Without numFocus=3
#      0     1   20.68772    0   31    0.51164   20.68772  3943%     -    1s
# * 1592   751              33       0.5168438   11.04052  2036%  10.7    1s
# H 1978   885                       0.6322989   10.26536  1523%  10.9    1s
# * 1983   885              32       0.6380561   10.26536  1509%  10.9    1s
# * 2758  1107              33       0.6423330    8.91701  1288%  11.1    1s

# Cutting planes:
#   Gomory: 29
#   Cover: 1
#   Implied bound: 5
#   MIR: 9
#   Flow cover: 13
#   RLT: 5
#   Relax-and-lift: 6

# Explored 3174 nodes (38233 simplex iterations) in 2.10 seconds (3.73 work units)
# Thread count was 8 (of 8 available processors)

# Solution count 5: 0.642333 0.638056 0.632299 ... 0.511639
# No other solutions better than 0.642333
# -------------------------------------------------------------


# -------------------------------------------------------------
# With numFocus=3
#      0     2 4359.17298    0   63    0.51164 4359.17298      -     -    0s
# * 2689  1439              99       0.5168438    5.85460  1033%  61.2    2s
# * 2690  1376              99       0.5175604    5.85460  1031%  61.2    2s
# H 3775  1396                       0.5325435    4.86679   814%  48.3    3s
# H 3776  1317                       0.6322989    4.86679   670%  48.3    3s
# H 3777  1262                       0.6380561    4.86679   663%  48.3    3s
# * 7709  1743              86       0.6423330    3.43489   435%  30.8    3s
# H12389  1599                       0.6423331    2.26531   253%  24.1    4s

# Explored 15305 nodes (327425 simplex iterations) in 4.61 seconds (13.33 work units)
# Thread count was 8 (of 8 available processors)

# Solution count 8: 0.642333 0.642333 0.638056 ... 0.511639
# No other solutions better than 0.642333
# -------------------------------------------------------------

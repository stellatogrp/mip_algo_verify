from time import perf_counter

import numpy as np
from gurobipy import GRB, Model, max_, quicksum


def SoftThresholding(y, lambda_t):
    if y > lambda_t:
        return y - lambda_t
    elif y < -lambda_t:
        return y + lambda_t
    return 0.0

def BoundPreprocessing(K, At, Bt, lambda_t, z_0, c_theta, r_theta):
    # Preprocess LB and UB with standard techniques
    n = len(z_0)
    m = len(c_theta)

    y_LB, y_UB = {}, {}
    z_LB, z_UB = {}, {}
    LB_theta, UB_theta = {}, {}

    for i in range(n):
        z_UB[i, 0] = z_0[i]
        z_LB[i, 0] = z_0[i]

        # Linear bound indpendent of the iteration k
        LB_theta[i] = sum(Bt[i, h]*(c_theta[h]-r_theta) for h in range(m) if Bt[i, h] > 0) + \
                      sum(Bt[i, h]*(c_theta[h]+r_theta) for h in range(m) if Bt[i, h] < 0)

        UB_theta[i] = sum(Bt[i, h]*(c_theta[h]+r_theta) for h in range(m) if Bt[i, h] > 0) + \
                      sum(Bt[i, h]*(c_theta[h]-r_theta) for h in range(m) if Bt[i, h] < 0)

    for k in range(1, K):
        for i in range(n):
            y_LB[i, k]  = LB_theta[i]
            y_LB[i, k] += sum(At[i, j]*z_LB[j, k-1] for j in range(n) if At[i, j] > 0)
            y_LB[i, k] += sum(At[i, j]*z_UB[j, k-1] for j in range(n) if At[i, j] < 0)

            y_UB[i, k] = UB_theta[i]
            y_UB[i, k] += sum(At[i, j]*z_UB[j, k-1] for j in range(n) if At[i, j] > 0)
            y_UB[i, k] += sum(At[i, j]*z_LB[j, k-1] for j in range(n) if At[i, j] < 0)

            if y_LB[i, k] > y_UB[i, k]:
                raise ValueError('Basic Infeasible bounds y', y_LB[i, k], y_UB[i, k])

            # TODO: REMOVE 5.75 and use the correct Delta bounding procedure
            z_LB[i, k] = max(z_0[i] - 5.75*k, SoftThresholding(y_LB[i, k], lambda_t))
            z_UB[i, k] = min(z_0[i] + 5.75*k, SoftThresholding(y_UB[i, k], lambda_t))

    return y_LB, y_UB, z_LB, z_UB


def BuildRelaxedModel(K, At, Bt, lambda_t, z_0, c_theta, r_theta, y_LB, y_UB, z_LB, z_UB):
    n = len(z_0)
    m = len(c_theta)

    model = Model()
    model.Params.OutputFlag = 0
    #model.Params.NumericFocus = 3
    # model.Params.PreSolve = 0
    # model.Params.DualReductions = 0
    # model.Params.FeasibilityTol = 1e-09
    # model.Params.Method = 0

    # Create variables
    z, y = {}, {}
    for i in range(n):
        z[i, 0] = model.addVar(lb=z_0[i], ub=z_0[i])

    for k in range(1, K):
        for i in range(n):
            if y_LB[i, k] > y_UB[i, k]:
                raise ValueError('Infeasible bounds y', i, k, y_LB[i, k], y_UB[i, k])
            if z_LB[i, k] > z_UB[i, k]:
                raise ValueError('Infeasible bounds z', i, k, z_LB[i, k], z_UB[i, k])

            z[i, k] = model.addVar(lb=z_LB[i, k], ub=z_UB[i, k])
            y[i, k] = model.addVar(lb=y_LB[i, k], ub=y_UB[i, k])

    theta = {}
    for h in range(m):
        theta[h] = model.addVar(lb=c_theta[h] - r_theta, ub=c_theta[h] + r_theta)

    model.update()

    # Constraints affine step
    for k in range(1, K):
        for i in range(n):
            model.addConstr(y[i, k] == quicksum(At[i,j]*z[j, k-1] for j in range(n)) + quicksum(Bt[i, h]*theta[h] for h in range(m)))

    # Relaxation of the soft-thresholding
    # NOTE: bounding on y, I stop at K-1, do not put contstraints on z[k]
    for k in range(1, K-1):
        for i in range(n):
            if y_LB[i, k] >= lambda_t:
                model.addConstr(z[i, k] == y[i, k] - lambda_t)

            elif y_UB[i, k] <= -lambda_t:
                model.addConstr(z[i, k] == y[i, k] + lambda_t)

            elif y_LB[i, k] >= -lambda_t and y_UB[i, k] <= lambda_t:
                model.addConstr(z[i, k] == 0.0)

            elif y_LB[i, k] < -lambda_t and y_UB[i, k] > lambda_t:
                model.addConstr(z[i, k] >= y[i, k] - lambda_t)
                model.addConstr(z[i, k] <= y[i, k] + lambda_t)

                model.addConstr(z[i, k] <= z_UB[i, k]/(y_UB[i, k] + lambda_t)*(y[i, k] + lambda_t))
                model.addConstr(z[i, k] >= z_LB[i, k]/(y_LB[i, k] - lambda_t)*(y[i, k] - lambda_t))

            elif -lambda_t <= y_LB[i, k] <= lambda_t and y_UB[i, k] > lambda_t:
                model.addConstr(z[i, k] >= 0)
                model.addConstr(z[i, k] <= z_UB[i, k]/(y_UB[i, k] - y_LB[i, k])*(y[i, k] - y_LB[i, k]))
                model.addConstr(z[i, k] >= y[i, k] - lambda_t)

            elif -lambda_t <= y_UB[i, k] <= lambda_t and y_LB[i, k] < -lambda_t:
                model.addConstr(z[i, k] <= 0)
                model.addConstr(z[i, k] >= z_LB[i, k]/(y_LB[i, k] - y_UB[i, k])*(y[i, k] - y_UB[i, k]))
                model.addConstr(z[i, k] <= y[i, k] + lambda_t)
            else:
                raise RuntimeError('Unreachable code', y_LB[i, k], y_UB[i, k], lambda_t)

    model.update()
    return model, y


def BoundTightY(K, At, Bt, lambda_t, z_0, c_theta, r_theta, basic=True):
    y_LB, y_UB, z_LB, z_UB = BoundPreprocessing(K, At, Bt, lambda_t, z_0, c_theta, r_theta)

    if basic:
        return y_LB, y_UB, z_LB, z_UB

    n = len(z_0)

    for k in range(1, K):
        model, y = BuildRelaxedModel(k+1, At, Bt, lambda_t, z_0, c_theta, r_theta, y_LB, y_UB, z_LB, z_UB)
        print('^^^^^^^^^^^^ Bound tighting, K =', k, '^^^^^^^^^^')
        for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
            for i in range(n):
                # Optimize for given variable
                model.setObjective(y[i, k], sense)
                model.update()

                model.optimize()

                if model.status != GRB.OPTIMAL:
                    print('bound tighting failes, GRB model status:', model.status)
                    return None

                # Update bounds
                obj = model.objVal
                if sense == GRB.MAXIMIZE:
                    y_UB[i, k] = obj
                    # TODO: REMOVE 5.75 and use the correct Delta bounding procedure
                    z_UB[i, k] = min(z_0[i] + 5.75*k, SoftThresholding(y_UB[i, k], lambda_t))
                else:
                    y_LB[i, k] = obj
                    # TODO: REMOVE 5.75 and use the correct Delta bounding procedure
                    z_LB[i, k] = max(z_0[i] - 5.75*k, SoftThresholding(y_LB[i, k], lambda_t))

                if y_LB[i, k] > y_UB[i, k]:
                    raise ValueError('Infeasible bounds y', sense, i, k, y_LB[i, k], y_UB[i, k])
                if z_LB[i, k] > z_UB[i, k]:
                    raise ValueError('Infeasible bounds z', sense, i, k, z_LB[i, k], z_UB[i, k])

    return y_LB, y_UB, z_LB, z_UB


def VerifyISTA_withBounds(K, pnorm, At, Bt, lambda_t, z_0, c_theta, r_theta, Deltas, y_LB, y_UB, z_LB, z_UB, zbar, ybar, thetabar):
    n = len(z_0)
    m = len(c_theta)

    model = Model()
    model.Params.OutputFlag = 0
    #model.Params.NumericFocus = 3
    #model.Params.PreSolve = 0
    #model.Params.DualReductions = 0
    #model.Params.FeasibilityTol = 1e-09

    # Create variables
    z, y = {}, {}
    up, un, v = {}, {}, {}

    for i in range(n):
        z[i, 0] = model.addVar(lb=z_0[i], ub=z_0[i])

    for k in range(1, K):
        for i in range(n):
            y[i,k] = model.addVar(lb=y_LB[i, k], ub=y_UB[i, k])
            z[i,k] = model.addVar(lb=z_LB[i, k], ub=z_UB[i, k])

    theta = {}
    for h in range(m):
        theta[h] = model.addVar(lb=c_theta[h]-r_theta, ub=c_theta[h]+r_theta)

    for i in range(n):
        # abs value for objective
        v[i] = model.addVar(vtype=GRB.BINARY)
        up[i] = model.addVar(lb=0, ub=max(0.0, z_UB[i, K-1] - z_LB[i, K-2]))
        un[i] = model.addVar(lb=0, ub=max(0.0, z_UB[i, K-2] - z_LB[i, K-1]))

    # Introduce these variables only when necessary
    w1, w2 = {}, {}

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
        model.addConstr(up[i] - un[i] == z[i, K-1] - z[i, K-2])
        model.addConstr(up[i] <= (z_UB[i, K-1] - z_LB[i, K-2])*v[i])
        model.addConstr(un[i] <= (z_UB[i, K-2] - z_LB[i, K-1])*(1 - v[i]))

    for k in range(1, K):
        for i in range(n):
            model.addConstr(y[i, k] == quicksum(At[i, j]*z[j, k-1] for j in range(n)) + quicksum(Bt[i,h]*theta[h] for h in range(m)))

    for k in range(1, K):
        for i in range(n):
            # box-bound constraints (polyhedral relaxation of soft-thresholding)

            if y_LB[i, k] >= lambda_t:
                model.addConstr(z[i, k] == y[i, k] - lambda_t)

            elif y_UB[i, k] <= -lambda_t:
                model.addConstr(z[i, k] == y[i, k] + lambda_t)

            elif y_LB[i, k] >= -lambda_t and y_UB[i, k] <= lambda_t:
                model.addConstr(z[i, k] == 0.0)

            else:
                if y_LB[i, k] < -lambda_t and y_UB[i, k] > lambda_t:
                    w1[i, k] = model.addVar(vtype=GRB.BINARY)
                    w2[i, k] = model.addVar(vtype=GRB.BINARY)
                    model.update()

                    model.addConstr(z[i, k] >= y[i, k] - lambda_t)
                    model.addConstr(z[i, k] <= y[i, k] + lambda_t)

                    model.addConstr(z[i, k] <= z_UB[i, k]/(y_UB[i, k] + lambda_t)*(y[i, k] + lambda_t))
                    model.addConstr(z[i, k] >= z_LB[i, k]/(y_LB[i, k] - lambda_t)*(y[i, k] - lambda_t))

                    # Upper right part: w1 = 1, y >= lambda_t
                    model.addConstr(z[i, k] <= y[i,k] - lambda_t + (lambda_t + z_UB[i, k] - y_LB[i, k])*(1-w1[i,k]))
                    model.addConstr(y[i, k] >= lambda_t + (y_LB[i, k] - lambda_t)*(1-w1[i, k]))
                    model.addConstr(y[i, k] <= lambda_t + (y_UB[i, k] - lambda_t)*w1[i, k])
                    # Lower left part: w2 = 1, y <= -lambda_t
                    model.addConstr(z[i, k] >= y[i,k] + lambda_t + (z_LB[i, k] + y_UB[i, k])*(1-w2[i,k]))
                    model.addConstr(y[i, k] <= -lambda_t + (y_UB[i, k] + lambda_t)*(1-w2[i, k]))
                    model.addConstr(y[i, k] >= -lambda_t + (y_LB[i, k] + lambda_t)*w2[i, k])
                    # The left and right part cannot be hold at the same time (improve LP relaxation)
                    model.addConstr(w1[i, k] + w2[i, k] <= 1)

                elif -lambda_t <= y_LB[i, k] <= lambda_t and y_UB[i, k] > lambda_t:
                    w1[i, k] = model.addVar(vtype=GRB.BINARY)
                    model.update()

                    model.addConstr(z[i, k] >= 0)
                    model.addConstr(z[i, k] <= z_UB[i, k]/(y_UB[i, k] - y_LB[i, k])*(y[i, k] - y_LB[i, k]))
                    model.addConstr(z[i, k] >= y[i, k] - lambda_t)

                    # Upper right part: w1 = 1, y >= lambda_t
                    model.addConstr(z[i, k] <= y[i,k] - lambda_t + (lambda_t + z_UB[i, k] - y_LB[i, k])*(1-w1[i,k]))
                    model.addConstr(y[i, k] >= lambda_t + (y_LB[i, k] - lambda_t)*(1-w1[i, k]))
                    model.addConstr(y[i, k] <= lambda_t + (y_UB[i, k] - lambda_t)*w1[i, k])

                elif -lambda_t <= y_UB[i, k] <= lambda_t and y_LB[i, k] < -lambda_t:
                    w2[i, k] = model.addVar(vtype=GRB.BINARY)
                    model.update()

                    model.addConstr(z[i, k] <= 0)
                    model.addConstr(z[i, k] >= z_LB[i, k]/(y_LB[i, k] - y_UB[i, k])*(y[i, k] - y_UB[i, k]))
                    model.addConstr(z[i, k] <= y[i, k] + lambda_t)

                    # Lower left part: w2 = 1, y <= -lambda_t
                    model.addConstr(z[i, k] >= y[i,k] + lambda_t + (z_LB[i, k] + y_UB[i, k])*(1-w2[i,k]))
                    model.addConstr(y[i, k] <= -lambda_t + (y_UB[i, k] + lambda_t)*(1-w2[i, k]))
                    model.addConstr(y[i, k] >= -lambda_t + (y_LB[i, k] + lambda_t)*w2[i, k])

    # Complete previous solution
    if zbar is not None:
        for i, k in zbar:
            z[i, k].Start = zbar[i,k]
        for h in thetabar:
            theta[h].Start = thetabar[h]

    model.update()

    # Solve the model
    model.optimize()

    # Check the status
    # TODO: add a time limit and check the status
    if model.status != GRB.OPTIMAL:
        print('model status:', model.status)
        for i in range(n):
            print('Bounds:', list(map(lambda x: round(x, 3), [y_LB[i, K], y_UB[i, K], z_LB[i, K], z_UB[i, K]])))

        return None

    return model.objVal, {(i,k): z[i,k].X for i, k in z}, {(i,k): y[i,k].X for i, k in y}, {j: theta[j].X for j in theta}


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

def run_ista(K_max, At, Bt, lambda_t, z_0, c_theta):
    # Initialize variables
    z2 = np.copy(z_0)
    nn = np.zeros(K_max)

    for i in range(K_max):
        z1 = np.copy(z2)
        y2 = At @ z1 + Bt@c_theta
        z2 = np.maximum(y2, lambda_t) - np.maximum(-y2, lambda_t) # Soft thresholding
        nn[i] = np.linalg.norm(z2 - z1, ord=1) #float('inf'))

    return nn

def SampleMax(Samples, K_max, At, Bt, lambda_t, z_0, c_theta):
    max_norm = np.zeros(K_max)

    for _ in range(Samples):
        theta_rnd = c_theta + np.random.uniform(low=-r_theta, high=r_theta, size=len(c_theta))
        max_norm = np.maximum(max_norm, run_ista(K_max, At, Bt, lambda_t, z_0, theta_rnd))

    # Dump for PgFPlots
    for i in range(K_max):
        print('({}, {})'.format(i+1, max_norm[i]), end=' ')

if __name__ == '__main__':
    At, Bt, lambda_t, c_z, c_theta, r_theta = MakeData()

    # Number of iterations
    K = 201
    pnorm = 1

    if False:
        # 1000 samples and max K = 60
        t0 = perf_counter()
        SampleMax(1000000, K-1, At, Bt, lambda_t, c_z, c_theta)
        print('time:', perf_counter() - t0)

    if True:
        t0 = perf_counter()
        # Basic or advanced bound tightening
        y_LB, y_UB, z_LB, z_UB = BoundTightY(K, At, Bt, lambda_t, c_z, c_theta, r_theta, basic=False)

        # Iterative
        Deltas = []
        zbar, ybar, thetabar = None, None, None
        for k in range(1, K):
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VerifyISTA_withBounds, K =', k, '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            delta_k, zbar, ybar, thetabar = VerifyISTA_withBounds(k+1, pnorm, At, Bt, lambda_t, c_z, c_theta, r_theta, Deltas, y_LB, y_UB, z_LB, z_UB, zbar, ybar, thetabar)
            Deltas.append(delta_k)
            # Dump for plotting later with pgfplots
            #print('Residual for pgfplots:', Deltas)

        for i, v in enumerate(Deltas):
            print('({}, {})'.format(i+1, v), end=' ')

        print('complete runtime:', perf_counter() - t0)

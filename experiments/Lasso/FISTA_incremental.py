from time import perf_counter

import numpy as np
from gurobipy import GRB, Model, quicksum


def SoftThresholding(y, lambda_t):
    if y > lambda_t:
        return y - lambda_t
    elif y < -lambda_t:
        return y + lambda_t
    return 0.0


def InitBoundZ(n, m, Bt, z_0, c_theta, r_theta):
    w_LB, w_UB = {}, {}
    z_LB, z_UB = {}, {}
    LB_theta, UB_theta = {}, {}

    for i in range(n):
        # Starting point for the optimization
        w_LB[i, 0] = z_0[i]
        w_UB[i, 0] = z_0[i]
        z_LB[i, 0] = z_0[i]
        z_UB[i, 0] = z_0[i]

        # Linear bound indpendent of the iteration k
        LB_theta[i] = sum(Bt[i, h]*(c_theta[h]-r_theta) for h in range(m) if Bt[i, h] > 0) + \
                    sum(Bt[i, h]*(c_theta[h]+r_theta) for h in range(m) if Bt[i, h] < 0)

        UB_theta[i] = sum(Bt[i, h]*(c_theta[h]+r_theta) for h in range(m) if Bt[i, h] > 0) + \
                    sum(Bt[i, h]*(c_theta[h]-r_theta) for h in range(m) if Bt[i, h] < 0)

    return z_LB, z_UB, w_LB, w_UB, LB_theta, UB_theta


def BoundPreprocessing(n, k, At, y_LB, y_UB, w_LB, w_UB, LB_theta, UB_theta):
    # New bounds for a single class of variables
    for i in range(n):
        y_LB[i, k]  = LB_theta[i]
        y_LB[i, k] += sum(At[i, j]*w_LB[j, k-1] for j in range(n) if At[i, j] > 0)
        y_LB[i, k] += sum(At[i, j]*w_UB[j, k-1] for j in range(n) if At[i, j] < 0)

        y_UB[i, k] = UB_theta[i]
        y_UB[i, k] += sum(At[i, j]*w_UB[j, k-1] for j in range(n) if At[i, j] > 0)
        y_UB[i, k] += sum(At[i, j]*w_LB[j, k-1] for j in range(n) if At[i, j] < 0)

        if y_LB[i, k] > y_UB[i, k]:
            raise ValueError('Basic Infeasible bounds y', y_LB[i, k], y_UB[i, k])

    return y_LB, y_UB


def BuildRelaxedModel(K, At, Bt, lambda_t, z_0, c_theta, r_theta, y_LB, y_UB, z_LB, z_UB, w_LB, w_UB, beta):
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
    z, y, w, theta = {}, {}, {}, {}

    for h in range(m):
        theta[h] = model.addVar(lb=c_theta[h] - r_theta, ub=c_theta[h] + r_theta)

    for k in range(1, K):
        for i in range(n):
            if y_LB[i, k] > y_UB[i, k]:
                raise ValueError('Infeasible bounds y', i, k, y_LB[i, k], y_UB[i, k])
            y[i, k] = model.addVar(lb=y_LB[i, k], ub=y_UB[i, k])

    for k in range(K-1):
        for i in range(n):
            if z_LB[i, k] > z_UB[i, k]:
                raise ValueError('Infeasible bounds z', i, k, z_LB[i, k], z_UB[i, k])
            z[i, k] = model.addVar(lb=z_LB[i, k], ub=z_UB[i, k])
            w[i, k] = model.addVar(lb=w_LB[i, k], ub=w_UB[i, k])

    model.update()

    # Constraints affine step
    for k in range(1, K):
        for i in range(n):
            model.addConstr(y[i, k] == quicksum(At[i,j]*w[j, k-1] for j in range(n)) + quicksum(Bt[i, h]*theta[h] for h in range(m)))

    # Relaxation of the soft-thresholding
    # NOTE: bounding on y, I stop at K-1, do not put contstraints on z[k]
    for k in range(1, K-1):
        for i in range(n):
            model.addConstr(w[i, k] == z[i, k] + (beta[k-1] - 1)/beta[k]*(z[i, k] - z[i, k-1]))

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


def BoundTightY(k, n, At, Bt, lambda_t, z_0, c_theta, r_theta, y_LB, y_UB, z_LB, z_UB, w_LB, w_UB, beta):
    model, y = BuildRelaxedModel(k+1, At, Bt, lambda_t, z_0, c_theta, r_theta, y_LB, y_UB, z_LB, z_UB, w_LB, w_UB, beta)
    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE]:
        for i in range(n):
            model.setObjective(y[i, k], sense)
            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print('bound tighting failes, GRB model status:', model.status)
                return None

            if sense == GRB.MAXIMIZE:
                y_UB[i, k] = model.objVal
            else:
                y_LB[i, k] = model.objVal

            if y_LB[i, k] > y_UB[i, k]:
                raise ValueError('Infeasible bounds', sense, i, k, y_LB[i, k], y_UB[i, k])

    return y_LB, y_UB


def IncrementalVerifierFISTA(K, At, Bt, lambda_t, z_0, c_theta, r_theta):

    def Init_model():
        model = Model()
        model.Params.OutputFlag = 0

        for h in range(m):
            theta[h] = model.addVar(lb=c_theta[h] - r_theta, ub=c_theta[h] + r_theta)

        for i in range(n):
            w[i, 0] = model.addVar(lb=z_0[i], ub=z_0[i])
            z[i, 0] = model.addVar(lb=z_0[i], ub=z_0[i])

        model.update()

        return model

    def ModelNextStep(model, n, k, At, Bt, lambda_t, z_0, c_theta, r_theta, y_LB, y_UB, z_LB, z_UB, beta):

        # Devo agginugere nuove variabili e vincoli
        for i in range(n):
            y[i, k] = model.addVar(lb=y_LB[i, k], ub=y_UB[i, k])

        for i in range(n):
            if (i, k) not in z_LB:
                z_LB[i, k] = SoftThresholding(y_LB[i, k], lambda_t)
                z_UB[i, k] = SoftThresholding(y_UB[i, k], lambda_t)

            z[i, k] = model.addVar(lb=z_LB[i, k], ub=z_UB[i, k])

            if k > 0:
                w_LB[i, k] = (1 + (beta[k-1]-1)/beta[k])*z_LB[i, k] - (beta[k-1]-1)/beta[k]*z_UB[i, k-1]
                w_UB[i, k] = (1 + (beta[k-1]-1)/beta[k])*z_UB[i, k] - (beta[k-1]-1)/beta[k]*z_LB[i, k-1]
                w[i, k] = model.addVar(lb=w_LB[i, k], ub=w_UB[i, k])

        if k == 1:
            for i in range(n):
                v[i] = model.addVar(vtype=GRB.BINARY)
                up[i] = model.addVar(lb=0, ub=max(0.0, z_UB[i, k] - z_LB[i, k-1]))
                un[i] = model.addVar(lb=0, ub=max(0.0, z_UB[i, k-1] - z_LB[i, k]))
            # Mannhattan norm (p=1)
            model.setObjective(quicksum((up[i] + un[i]) for i in range(n)), GRB.MAXIMIZE)
        else:
            for i in range(n):
                up[i].UB = max(0.0, z_UB[i, k] - z_LB[i, k-1])
                un[i].UB = max(0.0, z_UB[i, k-1] - z_LB[i, k])
                model.remove(objcnstr1[i])
                model.remove(objcnstr2[i])
                model.remove(objcnstr3[i])

            model.update()

        # Constraints fun obj
        for i in range(n):
            objcnstr1[i] = model.addConstr(up[i] - un[i] == z[i, k] - z[i, k-1])
            objcnstr2[i] = model.addConstr(up[i] <= (z_UB[i, k] - z_LB[i, k-1])*v[i])
            objcnstr3[i] = model.addConstr(un[i] <= (z_UB[i, k-1] - z_LB[i, k])*(1 - v[i]))

        model.update()

        for i in range(n):
            model.addConstr(y[i, k] == quicksum(At[i, j]*w[j, k-1] for j in range(n)) + quicksum(Bt[i, h]*theta[h] for h in range(m)))

        # TODO: se aggiorni i bounds per Z, dovrei rimuovere tutto questi vincoli per k-1 e rimmetterli con il bound aggiornato, più stringente
        for i in range(n):
            model.addConstr(w[i, k] == z[i, k] + (beta[k-1] - 1)/beta[k]*(z[i, k] - z[i, k-1]))

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
                    model.addConstr(z[i, k] <= y[i, k] - lambda_t + (lambda_t + z_UB[i, k] - y_LB[i, k])*(1-w1[i, k]))
                    model.addConstr(y[i, k] >= lambda_t + (y_LB[i, k] - lambda_t)*(1-w1[i, k]))
                    model.addConstr(y[i, k] <= lambda_t + (y_UB[i, k] - lambda_t)*w1[i, k])
                    # Lower left part: w2 = 1, y <= -lambda_t
                    model.addConstr(z[i, k] >= y[i, k] + lambda_t + (z_LB[i, k] + y_UB[i, k])*(1-w2[i, k]))
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
                    model.addConstr(z[i, k] <= y[i, k] - lambda_t + (lambda_t + z_UB[i, k] - y_LB[i, k])*(1-w1[i, k]))
                    model.addConstr(y[i, k] >= lambda_t + (y_LB[i, k] - lambda_t)*(1-w1[i, k]))
                    model.addConstr(y[i, k] <= lambda_t + (y_UB[i, k] - lambda_t)*w1[i, k])

                elif -lambda_t <= y_UB[i, k] <= lambda_t and y_LB[i, k] < -lambda_t:
                    w2[i, k] = model.addVar(vtype=GRB.BINARY)
                    model.update()

                    model.addConstr(z[i, k] <= 0)
                    model.addConstr(z[i, k] >= z_LB[i, k]/(y_LB[i, k] - y_UB[i, k])*(y[i, k] - y_UB[i, k]))
                    model.addConstr(z[i, k] <= y[i, k] + lambda_t)

                    # Lower left part: w2 = 1, y <= -lambda_t
                    model.addConstr(z[i, k] >= y[i, k] + lambda_t + (z_LB[i, k] + y_UB[i, k])*(1-w2[i, k]))
                    model.addConstr(y[i, k] <= -lambda_t + (y_UB[i, k] + lambda_t)*(1-w2[i, k]))
                    model.addConstr(y[i, k] >= -lambda_t + (y_LB[i, k] + lambda_t)*w2[i, k])
                else:
                    raise RuntimeError('Unreachable code', y_LB[i, k], y_UB[i, k], lambda_t)

        # Risolve e restituisco il nuovo delta_k
        model.optimize()

        # Check the status
        # TODO: add a time limit and check the status
        if model.status != GRB.OPTIMAL:
            print('model verifier status:', model.status)
            return None

        return model.objVal


    # Beta_k once for all
    beta = {}
    beta[0] = 1.0
    for k in range(1, K):
        beta[k] = (1 + np.sqrt(1 + 4*beta[k-1]**2))/2

    # Space dimensions
    n = len(z_0)
    m = len(c_theta)
    # Local variables
    z_LB, z_UB, w_LB, w_UB, LB_theta, UB_theta = InitBoundZ(n, m, Bt, z_0, c_theta, r_theta)
    y_LB, y_UB = {}, {}

    # MIP model
    z, y, w, theta = {}, {}, {}, {}
    up, un, v = {}, {}, {}
    w1, w2 = {}, {}
    delta = {}
    objcnstr1, objcnstr2, objcnstr3 = {}, {}, {}

    model = Init_model()

    # Start iterating
    for k in range(1, K):
        y_LB, y_UB = BoundPreprocessing(n, k, At, y_LB, y_UB, w_LB, w_UB, LB_theta, UB_theta)
        y_LB, y_UB = BoundTightY(k, n, At, Bt, lambda_t, z_0, c_theta, r_theta, y_LB, y_UB, z_LB, z_UB, w_LB, w_UB, beta)

        result = ModelNextStep(model, n, k, At, Bt, lambda_t, z_0, c_theta, r_theta, y_LB, y_UB, z_LB, z_UB, beta)
        if result is None:
            return None
        delta[k] = result

        print(k, 'Delta:', delta[k], model.runtime)

        # Update bounds for z
        Dk = sum(delta[k] for k in delta)
        for i in range(n):
            if z_0[i] - Dk > SoftThresholding(y_LB[i, k], lambda_t):
                print('LB:', z_0[i] - Dk, SoftThresholding(y_LB[i, k], lambda_t))
            if z_0[i] + Dk < SoftThresholding(y_UB[i, k], lambda_t):
                print('UB:', z_0[i] + Dk, SoftThresholding(y_UB[i, k], lambda_t))

            z_LB[i, k] = max(z_0[i] - Dk, SoftThresholding(y_LB[i, k], lambda_t))
            z_UB[i, k] = min(z_0[i] + Dk, SoftThresholding(y_UB[i, k], lambda_t))
            z[i, k].LB = z_LB[i, k]
            z[i, k].UB = z_UB[i, k]
            model.update()

    return delta

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
    At, Bt, lambda_t, c_z, c_theta, r_theta = MakeData(True)
    K = 200

    t0 = perf_counter()
    deltas = IncrementalVerifierFISTA(K+1, At, Bt, lambda_t, c_z, c_theta, r_theta)
    print('Elapsed time:', perf_counter() - t0)
    print('Deltas:', deltas)

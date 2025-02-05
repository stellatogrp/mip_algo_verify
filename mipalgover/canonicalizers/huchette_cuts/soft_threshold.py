import copy

import numpy as np


def st_add_pos_conv_cuts(wi, ai, y, lambd, y_l, y_u):
    L_hat = np.zeros(y_l.shape)
    U_hat = np.zeros(y_u.shape)

    for j in range(y.shape[0]):
        if ai[j] >= 0:
            L_hat[j] = y_l[j]
            U_hat[j] = y_u[j]
        else:
            L_hat[j] = y_u[j]
            U_hat[j] = y_l[j]

    Iint, rhs, lI, h = compute_v_pos(ai, y, lambd, L_hat, U_hat)

    if Iint is None:
        return None, None, None, None, None

    if wi > rhs + 1e-6:
        return Iint, lI, h, L_hat, U_hat
    else:
        return None, None, None, None, None


def compute_v_pos(wi, xi, lambd, Lhat, Uhat):
    idx = np.arange(wi.shape[0])

    filtered_idx = np.array([j for j in idx if wi[j] != 0 and np.abs(Uhat[j] - Lhat[j]) > 1e-7])

    def key_func(j):
        return (xi[j] - Lhat[j]) / (Uhat[j] - Lhat[j])

    keys = np.array([key_func(j) for j in filtered_idx])
    sorted_idx = np.argsort(keys)  # this is nondecreasing, should be the corrct one according to paper
    # sorted_idx = np.argsort(keys)[::-1]
    filtered_idx = filtered_idx[sorted_idx]

    I = np.array([])
    Icomp = set(range(wi.shape[0]))

    lI = compute_lI(wi, xi, lambd, Lhat, Uhat, I, np.array(list(Icomp)))
    if lI < 0:
        return None, None, None, None

    for h in filtered_idx:
        Itest = np.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        lI_new = compute_lI(wi, xi, lambd, Lhat, Uhat, Itest.astype(np.int32), np.array(list(Icomp_test)))
        if lI_new < 0:
            Iint = I.astype(np.int32)
            rhs = np.sum(np.multiply(wi[Iint], xi[Iint])) + lI / (Uhat[int(h)] - Lhat[int(h)]) * (xi[int(h)] - Lhat[int(h)])
            return Iint, rhs, lI, int(h)

        I = Itest
        Icomp = Icomp_test
        lI = lI_new
    else:
        return None, None, None, None


def compute_lI(w, x, lambd, Lhat, Uhat, I, Icomp):
    if I.shape[0] == 0:
        return np.sum(np.multiply(w, Uhat)) - lambd
    if Icomp.shape[0] == 0:
        return np.sum(np.multiply(w, Lhat)) - lambd

    w_I = w[I]
    w_Icomp = w[Icomp]

    Lhat_I = Lhat[I]
    Uhat_I = Uhat[Icomp]

    return np.sum(np.multiply(w_I, Lhat_I)) + np.sum(np.multiply(w_Icomp, Uhat_I)) - lambd


def st_create_new_pos_constr(w, a, y, lambd, Iint, lI, h, Lhat, Uhat):
    new_constr = 0
    for idx in Iint:
        new_constr += a[idx] * (y[idx] - Lhat[idx])
    # new_constr += (lI - lambd) / (Uhat[h] - Lhat[h]) * (y[h] - Lhat[h])
     # this lI func already factors in lambd so we dont subtract it again from the top
    new_constr += lI / (Uhat[h] - Lhat[h]) * (y[h] - Lhat[h])

    return w <= new_constr

# below are the negative side constraints

def st_add_neg_conv_cuts(wi, ai, y, lambd, y_l, y_u):
    L_hat = np.zeros(y_l.shape)
    U_hat = np.zeros(y_u.shape)

    for j in range(y.shape[0]):
        if ai[j] >= 0:
            L_hat[j] = y_l[j]
            U_hat[j] = y_u[j]
        else:
            L_hat[j] = y_u[j]
            U_hat[j] = y_l[j]

    Iint, rhs, uI, h = compute_v_neg(ai, y, lambd, L_hat, U_hat)

    if Iint is None:
        return None, None, None, None, None

    if wi < rhs - 1e-6:
        return Iint, uI, h, L_hat, U_hat
    else:
        return None, None, None, None, None

def compute_v_neg(wi, xi, lambd, Lhat, Uhat):
    idx = np.arange(wi.shape[0])
    filtered_idx = np.array([j for j in idx if wi[j] != 0 and np.abs(Lhat[j] - Uhat[j]) > 1e-7])

    def key_func(j):
        return (xi[j] - Uhat[j]) / (Lhat[j] - Uhat[j])

    keys = np.array([key_func(j) for j in filtered_idx])
    sorted_idx = np.argsort(keys)  # should this be in reverse order?
    # sorted_idx = np.argsort(keys)[::-1]
    filtered_idx = filtered_idx[sorted_idx]

    I = np.array([])
    Icomp = set(range(wi.shape[0]))

    uI = compute_uI(wi, xi, lambd, Lhat, Uhat, I, np.array(list(Icomp)))

    if uI > 0:
        return None, None, None, None

    for h in filtered_idx:
        Itest = np.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        uI_new = compute_uI(wi, xi, lambd, Lhat, Uhat, Itest.astype(np.int32), np.array(list(Icomp_test)))
        if uI_new > 0:
            Iint = I.astype(np.int32)
            rhs = np.sum(np.multiply(wi[Iint], xi[Iint])) + uI / (Lhat[int(h)] - Uhat[int(h)]) * (xi[int(h)] - Uhat[int(h)])
            return Iint, rhs, uI, int(h)

        I = Itest
        Icomp = Icomp_test
        uI = uI_new
    else:
        return None, None, None, None


def compute_uI(w, x, lambd, Lhat, Uhat, I, Icomp):
    if I.shape[0] == 0:
        return np.sum(np.multiply(w, Lhat)) + lambd
    if Icomp.shape[0] == 0:
        return np.sum(np.multiply(w, Uhat)) + lambd

    w_I = w[I]
    w_Icomp = w[Icomp]

    Lhat_I = Lhat[Icomp]
    Uhat_I = Uhat[I]

    return np.sum(np.multiply(w_I, Uhat_I)) + np.sum(np.multiply(w_Icomp, Lhat_I)) + lambd


def st_create_new_neg_constr(w, a, y, lambd, Iint, uI, h, Lhat, Uhat):
    new_constr = 0

    # for idx in Iint:
    #     new_constr += a[idx] * (y[idx] - Uhat[idx])
    # print(uI)
    # print(-lambd)
    # print(Iint)
    # print(h)
    # new_constr += (uI) / (Lhat[h] - Uhat[h]) * (y[h] - Uhat[h])
    # return w >= new_constr

    for idx in Iint:
        # new_constr += a[idx] * (Uhat[idx] - y[idx])
        new_constr += a[idx] * (y[idx] - Uhat[idx])
    # new_constr += (uI + lambd) / (Uhat[h] - Lhat[h]) * (Uhat[h] - y[h])
    new_constr += uI / (Uhat[h] - Lhat[h]) * (Uhat[h] - y[h])
    return w >= new_constr

import copy

import numpy as np


def relu_create_new_constr(w, a, y, Iint, lI, h, Lhat, Uhat):
    new_constr = 0
    for idx in Iint:
        new_constr += a[idx] * (y[idx] - Lhat[idx])
    new_constr += lI / (Uhat[h] - Lhat[h]) * (y[h] - Lhat[h])

    return w <= new_constr

def compute_lI(w, x, b, Lhat, Uhat, I, Icomp):
    if I.shape[0] == 0:
        return np.sum(np.multiply(w, Uhat)) + b
    if Icomp.shape[0] == 0:
        return np.sum(np.multiply(w, Lhat)) + b

    w_I = w[I]
    w_Icomp = w[Icomp]

    Lhat_I = Lhat[I]
    Uhat_I = Uhat[Icomp]

    return np.sum(np.multiply(w_I, Lhat_I)) + np.sum(np.multiply(w_Icomp, Uhat_I)) + b


def compute_v(wi, xi, b, Lhat, Uhat):
    idx = np.arange(wi.shape[0])
    # log.info(idx)

    filtered_idx = np.array([j for j in idx if wi[j] != 0 and np.abs(Uhat[j] - Lhat[j]) > 1e-7])
    # log.info(filtered_idx)

    def key_func(j):
        return (xi[j] - Lhat[j]) / (Uhat[j] - Lhat[j])

    keys = np.array([key_func(j) for j in filtered_idx])
    # log.info(keys)
    sorted_idx = np.argsort(keys)
    filtered_idx = filtered_idx[sorted_idx]

    # log.info(filtered_idx)

    I = np.array([])
    Icomp = set(range(wi.shape[0]))

    # log.info(Icomp)

    lI = compute_lI(wi, xi, b, Lhat, Uhat, I, np.array(list(Icomp)))
    # log.info(f'original lI: {lI}')
    # print(f'original lI: {lI}')
    if lI < 0:
        return None, None, None, None

    for h in filtered_idx:
        Itest = np.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        # log.info(Itest)
        # log.info(Icomp_test)
        # print(Itest)
        # print(Icomp_test)

        lI_new = compute_lI(wi, xi, b, Lhat, Uhat, Itest.astype(np.int32), np.array(list(Icomp_test)))
        # log.info(lI_new)
        if lI_new < 0:
            Iint = I.astype(np.int32)
            # log.info(f'h={h}')
            # log.info(f'lI before and after: {lI}, {lI_new}')
            # print(f'h={h}')
            # print(f'lI before and after: {lI}, {lI_new}')
            rhs = np.sum(np.multiply(wi[Iint], xi[Iint])) + lI / (Uhat[int(h)] - Lhat[int(h)]) * (xi[int(h)] - Lhat[int(h)])
            return Iint, rhs, lI, int(h)

        I = Itest
        Icomp = Icomp_test
        lI = lI_new
    else:
        return None, None, None, None


def relu_add_conv_cuts(wi, ai, y, y_l, y_u):
    L_hat = np.zeros(y_l.shape)
    U_hat = np.zeros(y_u.shape)

    for j in range(y.shape[0]):
        if ai[j] >= 0:
            L_hat[j] = y_l[j]
            U_hat[j] = y_u[j]
        else:
            L_hat[j] = y_u[j]
            U_hat[j] = y_l[j]

    Iint, rhs, lI, h = compute_v(ai, y, 0, L_hat, U_hat)

    if Iint is None:
        return None, None, None, None, None

    if wi > rhs + 1e-6:
        return Iint, lI, h, L_hat, U_hat
    else:
        return None, None, None, None, None

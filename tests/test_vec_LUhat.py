import numpy as np


def cond(pred, true_fun, false_fun, *operands):
    # to replicate jax behavior
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def test_vec_LU_hat():
    # testing that the vectorized form of
    np.random.seed(0)
    K = 1
    n = 10

    LB = -np.random.uniform(size=(K+1, n))
    UB = np.random.uniform(size=(K+1, n))

    A = np.random.normal(size=(n, n))

    L_hatA = np.zeros((K+1, n, n))
    U_hatA = np.zeros((K+1, n, n))

    for k in range(1, K+1):  # TODO: figure out how to vectorize this
        for i in range(n):  # i is the output indices of a relu
            for j in range(n):  # j in the input indices of a relu
                if A[i, j] >= 0:
                    # L_hatA = L_hatA.at[k, i, j].set(z_LB[k-1, j])
                    # U_hatA = U_hatA.at[k, i, j].set(z_UB[k-1, j])
                    L_hatA[k, i, j] = LB[k-1, j]
                    U_hatA[k, i, j] = UB[k-1, j]
                else:
                    # L_hatA = L_hatA.at[k, i, j].set(z_UB[k-1, j])
                    # U_hatA = U_hatA.at[k, i, j].set(z_LB[k-1, j])
                    L_hatA[k, i, j] = UB[k-1, j]
                    U_hatA[k, i, j] = LB[k-1, j]

    L_hatA_vec = np.zeros((K+1, n, n))
    U_hatA_vec = np.zeros((K+1, n, n))
    A_nonneg = np.where(A >= 0)
    A_neg = np.where(A < 0)

    for k in range(1, K+1):
        L_hatA_vec[k][A_nonneg] = LB[k-1][A_nonneg[1]]
        L_hatA_vec[k][A_neg] = UB[k-1][A_neg[1]]

        U_hatA_vec[k][A_nonneg] = UB[k-1][A_nonneg[1]]
        U_hatA_vec[k][A_neg] = LB[k-1][A_neg[1]]

    assert np.allclose(L_hatA, L_hatA_vec) and np.allclose(U_hatA, U_hatA_vec)

import numpy as np
import scipy.sparse as spa

from mipalgover.linexpr import LinExpr
from mipalgover.vector import Vector


def test_LinExpr_dense():
    np.random.seed(0)
    m = 15
    n = 10
    x = LinExpr(n)
    assert np.linalg.norm(x.decomposition_dict[x] - np.eye(n)) <= 1e-10
    y = Vector(n)
    z = x + y
    assert x in z.decomposition_dict
    assert y in z.decomposition_dict

    A = np.random.normal(size=(n, n))

    Ax = A @ x
    assert np.linalg.norm(Ax.decomposition_dict[x] - A) <= 1e-8

    negAx = -Ax
    assert np.linalg.norm(negAx.decomposition_dict[x] + A) <= 1e-8

    Axhalf = Ax / 2
    assert np.linalg.norm(Axhalf.decomposition_dict[x] - A / 2) <= 1e-8

    Axhalfmult = Ax * .5
    assert np.linalg.norm(Axhalfmult.decomposition_dict[x] - Axhalf.decomposition_dict[x]) <= 1e-8

    z2 = x - y
    assert x in z2.decomposition_dict
    assert y in z2.decomposition_dict

    null = x - x
    assert x not in null.decomposition_dict

    B = np.random.normal(size=(m, n))
    y = B @ x

    assert y.n == m
    assert not y.is_leaf


def test_LinExpr_sparse():
    m, n = 2, 3
    x = Vector(n)
    A = spa.random(m, n, density=0.4, random_state=10)
    y = Vector(m)

    test_expr = 5 * y + 2 * A @ x
    assert spa.linalg.norm(test_expr.decomposition_dict[y] - 5 * spa.eye(m)) <= 1e-10
    assert spa.linalg.norm(test_expr.decomposition_dict[x] - 2 * A) <= 1e-10

    test_expr2 = 3 * x - 4 * A.T @ y
    assert spa.linalg.norm(test_expr2.decomposition_dict[x] - 3 * spa.eye(n)) <= 1e-10
    assert spa.linalg.norm(test_expr2.decomposition_dict[y] + 4 * A.T) <= 1e-10


def test_composite_linexprs():
    m, n = 2, 3
    t = 0.1
    np.random.seed(0)

    # x_test = np.random.uniform(size=(n,))
    A = np.random.normal(size=(m, n))
    # c = np.random.uniform(size=(n,))
    # b = A @ x_test

    x = Vector(n)
    y = Vector(m)

    c_param = Vector(n)
    # b_param = Vector(n)

    test_expr = x - t * (c_param - A.T @ y)
    # print(test_expr.decomposition_dict)

    assert spa.linalg.norm(test_expr.decomposition_dict[x] - spa.eye(n)) <= 1e-10
    assert np.linalg.norm(test_expr.decomposition_dict[y] - t * A.T) <= 1e-10
    assert spa.linalg.norm(test_expr.decomposition_dict[c_param] + t * spa.eye(n)) <= 1e-10

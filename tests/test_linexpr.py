import numpy as np

from mipalgover.linexpr import LinExpr, Vector


def test_LinExpr_dense():
    np.random.seed(0)
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


def test_LinExpr_sparse():
    # TODO: add tests with sparse A matrix
    pass

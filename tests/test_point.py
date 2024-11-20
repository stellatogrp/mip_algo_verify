import numpy as np

from mipalgover.point import Point


def test_point_add():
    np.random.seed(0)
    n = 10
    x = Point(n)
    assert np.linalg.norm(x.decomposition_dict[x] - np.eye(n)) <= 1e-10
    y = Point(n)
    z = x + y
    assert x in z.decomposition_dict
    assert y in z.decomposition_dict

    A = np.random.normal(size=(n, n))
    # A = 2 * spa.eye(n)

    Ax = A @ x
    assert np.linalg.norm(Ax.decomposition_dict[x] - A) <= 1e-8

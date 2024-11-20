import numpy as np
from scipy.optimize import nnls


def test_placeholder():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    x, rnorm = nnls(A, b)
    assert np.allclose(np.dot(A, x), b)

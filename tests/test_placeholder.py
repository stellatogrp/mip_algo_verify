# Create symple pytest example

import numpy as np
from scipy.optimize import nnls


def test_nnls():
    # TODO: Change it this is just a placeholder
    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    x, rnorm = nnls(A, b)
    assert np.allclose(np.dot(A, x), b)

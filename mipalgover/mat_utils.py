import numpy as np
import scipy.sparse as spa


def is_valid_mat(mat):
    return isinstance(mat, np.ndarray) or spa.issparse(mat)

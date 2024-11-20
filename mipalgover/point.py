import numpy as np

from mipalgover.dict_utils import merge_dict, prune_dict

"""
Most of the code structure is borrowed from PEPit https://github.com/PerformanceEstimation/PEPit
"""


class Point(object):
    counter = 0
    list_of_leaf_points = list() # TODO: figure out if we need these

    def __init__(self,
                 n,
                 is_leaf=True,
                 decomposition_dict=None):
        self.is_leaf = is_leaf
        self.value = None
        self.n = n
        if is_leaf:
            assert decomposition_dict is None
            self.decomposition_dict = {self: np.eye(n)}
            self.counter = Point.counter
            Point.counter += 1
            Point.list_of_leaf_points.append(self)
        else:
            assert isinstance(decomposition_dict, dict)
            self.decomposition_dict = decomposition_dict
            self.counter = None

    def __add__(self, other):
        assert isinstance(other, Point)
        assert self.n == other.n

        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        merged_decomposition_dict = prune_dict(merged_decomposition_dict)

        return Point(self.n, is_leaf=False, decomposition_dict=merged_decomposition_dict)

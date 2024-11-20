import scipy.sparse as spa

from mipalgover.dict_utils import merge_dict, prune_dict
from mipalgover.mat_utils import is_valid_mat

"""
Most of the code structure is borrowed from PEPit https://github.com/PerformanceEstimation/PEPit
"""


class LinExpr(object):
    counter = 0
    list_of_leaf_LinExprs = list()  # TODO: figure out if we need these
    __array_priority__ = 1000  # this is necessary else numpy will not even look at the @ operator due to precedence

    def __init__(self,
                 n,
                 is_leaf=True,
                 decomposition_dict=None):
        self.is_leaf = is_leaf
        self.value = None
        self.n = n
        if is_leaf:
            assert decomposition_dict is None
            self.decomposition_dict = {self: spa.eye(n)}
            self.counter = LinExpr.counter
            LinExpr.counter += 1
            LinExpr.list_of_leaf_LinExprs.append(self)

            assert self.decomposition_dict[self].shape[1] == n
        else:
            assert isinstance(decomposition_dict, dict)
            self.decomposition_dict = decomposition_dict
            self.counter = None

    def __add__(self, other):
        assert isinstance(other, LinExpr)
        assert self.n == other.n

        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        merged_decomposition_dict = prune_dict(merged_decomposition_dict)

        return LinExpr(self.n, is_leaf=False, decomposition_dict=merged_decomposition_dict)

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            new_decomposition_dict = dict()
            for key, value in self.decomposition_dict.items():
                new_decomposition_dict[key] = other * value
            return LinExpr(self.n, is_leaf=False, decomposition_dict=new_decomposition_dict)
        else:
            raise TypeError(f'LinExprs can only be multiplied by scalar values.'
                            f'Got {type(other)}. If intending to use a matrix, use the @ symbol with the matrix on the left')

    def __mul__(self, other):
        return self.__rmul__(other=other)

    def __truediv__(self, other):
        return self.__rmul__(1 / other)

    def __neg__(self):
        return self.__rmul__(other=-1)

    def __sub__(self, other):
        return self.__add__(-other)

    def __matmul__(self, other):
        # return self.__rmatmul__(other=other)
        raise NotImplementedError('np matrix or sp matrix should be on the left of the @')

    def __rmatmul__(self, other):
        # print('__rmatmul__ being called')
        if is_valid_mat(other):
            new_decomposition_dict = dict()
            for key, value in self.decomposition_dict.items():
                new_decomposition_dict[key] = other @ value
            return LinExpr(other.shape[0], is_leaf=False, decomposition_dict=new_decomposition_dict)
        raise NotImplementedError(f'Object on the left should be np or sp.sparse matrix. Got {type(other)}')

    def eval(self):
        raise NotImplementedError('LinExpr.eval() needs to be implemented.')


class Vector(LinExpr):

    def __init__(self, n):
        return super().__init__(n)

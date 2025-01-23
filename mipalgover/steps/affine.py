import numpy as np

from mipalgover.linexpr import LinExpr
from mipalgover.steps.step import Step


class AffineStep(Step):

    def __init__(self, lhs_expr, rhs_expr, lhs_mat=None, lhs_mat_factorization=None):
        # self.lhs_expr = lhs_expr
        # self.rhs_expr = rhs_expr
        super().__init__(lhs_expr, rhs_expr)
        self.lhs_mat = lhs_mat
        if lhs_mat_factorization is None:
            # TODO: implement this
            raise NotImplementedError('not implemented when factorization missing')
            # self.lhs_mat_factorization = spa.linalg.factorized(lhs_expr)
        else:
            self.lhs_mat_factorization = lhs_mat_factorization

    def Minv_rhs(self):
        new_decomp_dict = {}
        for key, val in self.rhs_expr.decomposition_dict.items():
            Minv_val = np.linalg.solve(self.lhs_mat, val.todense())
            # print(type(Minv_val))
            # print(type(np.squeeze(np.asarray(Minv_val))))
            new_decomp_dict[key] = np.squeeze(np.asarray(Minv_val))
        return LinExpr(self.lhs_expr.get_output_dim(), is_leaf=False, decomposition_dict=new_decomp_dict)

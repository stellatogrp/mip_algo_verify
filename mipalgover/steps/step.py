import numpy as np


class Step(object):

    def __init__(self, lhs_expr, rhs_expr):
        self.lhs_expr = lhs_expr
        self.rhs_expr = rhs_expr
        self.rhs_lb = None
        self.rhs_ub = None

    def update_rhs_lb(self, new_rhs_lb):
        if self.rhs_lb is None:
            self.rhs_lb = new_rhs_lb
        else:
            old_rhs_lb = self.rhs_lb
            self.rhs_lb = np.maximum(old_rhs_lb, new_rhs_lb)

    def update_rhs_ub(self, new_rhs_ub):
        if self.rhs_ub is None:
            self.rhs_ub = new_rhs_ub
        else:
            old_rhs_ub = self.rhs_ub
            self.rhs_ub = np.minimum(old_rhs_ub, new_rhs_ub)

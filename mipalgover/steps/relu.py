from mipalgover.steps.step import Step


class ReluStep(Step):

    def __init__(self, lhs_expr, rhs_expr):
        # self.lhs_expr = lhs_expr
        # self.rhs_expr = rhs_expr
        # self.rhs_lb = None
        # self.rhs_ub = None
        super().__init__(lhs_expr, rhs_expr)

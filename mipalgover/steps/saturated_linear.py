from mipalgover.steps.step import Step


class SaturatedLinearStep(Step):

    def __init__(self, lhs_expr, rhs_expr, l, u):
        super().__init__(lhs_expr, rhs_expr)
        self.l = l
        self.u = u

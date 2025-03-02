from mipalgover.steps.step import Step


class SaturatedLinearStep(Step):

    def __init__(self, lhs_expr, rhs_expr, l, u, relax_binary_vars=False):
        super().__init__(lhs_expr, rhs_expr, relax_binary_vars=relax_binary_vars)
        self.l = l
        self.u = u

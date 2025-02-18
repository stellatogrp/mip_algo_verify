from mipalgover.steps.step import Step


class SoftThresholdStep(Step):

    def __init__(self, lhs_expr, rhs_expr, lambd, relax_binary_vars=False):
        super().__init__(lhs_expr, rhs_expr, relax_binary_vars=relax_binary_vars)
        self.lambd = lambd
        self.idx_with_right_binary_vars = set([])
        self.idx_with_left_binary_vars = set([])

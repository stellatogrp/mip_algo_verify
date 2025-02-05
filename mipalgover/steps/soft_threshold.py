from mipalgover.steps.step import Step


class SoftThresholdStep(Step):

    def __init__(self, lhs_expr, rhs_expr, lambd):
        super().__init__(lhs_expr, rhs_expr)
        self.lambd = lambd
        self.idx_with_right_binary_vars = set([])
        self.idx_with_left_binary_vars = set([])

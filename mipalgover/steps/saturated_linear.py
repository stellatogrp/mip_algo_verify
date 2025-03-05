from mipalgover.steps.step import Step


class SaturatedLinearStep(Step):

    def __init__(self, lhs_expr, rhs_expr, l, u, relax_binary_vars=False, equality_ranges=None):
        super().__init__(lhs_expr, rhs_expr, relax_binary_vars=relax_binary_vars)
        self.l = l
        self.u = u


        self.equality_ranges = equality_ranges

        if equality_ranges is None:
            self.equality_ranges = [(0, 0)]

        self.process_equality_ranges()

    def process_equality_ranges(self):
        # equality_ranges = self.equality_ranges

        ineq_indices = set(list(range(0, self.rhs_expr.get_output_dim())))
        eq_indices = set([])

        for curr_range in self.equality_ranges:
            curr_range_set = set(list(range(curr_range[0], curr_range[1])))

            ineq_indices -= curr_range_set
            eq_indices = eq_indices.union(curr_range_set)

        self.eq_indices = eq_indices
        self.ineq_indices = ineq_indices

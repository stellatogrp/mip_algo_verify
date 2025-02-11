from mipalgover.steps.step import Step


class ReluStep(Step):

    def __init__(self, lhs_expr, rhs_expr, proj_ranges=None, relax_binary_vars=False):
        # self.lhs_expr = lhs_expr
        # self.rhs_expr = rhs_expr
        # self.rhs_lb = None
        # self.rhs_ub = None
        super().__init__(lhs_expr, rhs_expr, relax_binary_vars=relax_binary_vars)
        self.proj_ranges = proj_ranges
        self._process_proj_ranges()
        self.idx_with_binary_vars = set([])

    def _process_proj_ranges(self):
        nonproj_indices = set(list(range(0, self.rhs_expr.get_output_dim())))
        proj_indices = set([])

        for curr_range in self.proj_ranges:
            curr_range_set = set(list(range(curr_range[0], curr_range[1])))

            nonproj_indices -= curr_range_set
            proj_indices = proj_indices.union(curr_range_set)

        self.nonproj_indices = nonproj_indices
        self.proj_indices = proj_indices

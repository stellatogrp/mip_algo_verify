from mipalgover.steps.step import Step
import numpy as np


class LinProStep(Step):
    def __init__(self, x_expr, c, A, b, M, N, relax_binary_vars=False):
        """
        Initialize LinProStep for the linear program:
        min c^T x
        s.t. Ax <= b

        KKT conditions with MILP formulation:
        - A^T y + c = 0 (dual feasibility)
        - y >= 0 (dual feasibility)
        - Ax <= b (primal feasibility)
        - Complementary slackness with binary variables:
          (b-Ax)_i <= M*w_i
          y_i <= N*(1-w_i)

        """
        super().__init__(x_expr, None, relax_binary_vars=relax_binary_vars)
        self.c = c
        self.A = A
        self.b = b
        
        m = b.shape[0]
        if np.isscalar(M):
            M_vec = np.full(m, M)
        else:
            M_vec = np.asarray(M).flatten()
            assert M_vec.shape[0] == m, "M has wrong length"

        if np.isscalar(N):
            N_vec = np.full(m, N)
        else:
            N_vec = np.asarray(N).flatten()
            assert N_vec.shape[0] == m, "N has wrong length"

        self.M = M_vec
        self.N = N_vec
        self.idx_with_binary_vars = set(range(b.shape[0]))  
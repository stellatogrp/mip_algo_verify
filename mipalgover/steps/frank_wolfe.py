from mipalgover.steps.step import Step
from mipalgover.vector import Vector


class FrankWolfeStep(Step):
    def __init__(self, lhs_expr, rhs_expr, P, q, A, b, alpha, M=None, relax_binary_vars=False):

        super().__init__(lhs_expr, rhs_expr, relax_binary_vars=relax_binary_vars)
        
        # Problem data
        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.alpha = alpha
        self.M = M
        
        #  auxiliary variables
        self.s = Vector(rhs_expr.n)  # vertex of feasible set
        self.y = Vector(A.shape[0])  # dual variables
        self.w = Vector(A.shape[0])  # binary variables for complementarity
        
       
      
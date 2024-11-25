import numpy as np

from mipalgover.canonicalizers.gurobi_canonicalizer import GurobiCanonicalizer
from mipalgover.steps.relu import ReluStep
from mipalgover.vector import Vector


class Verifier(object):

    def __init__(self,
                 num_obbt=3,
                 postprocess=False,
                 solver='gurobi'):

        self.num_obbt = num_obbt
        self.postprocess = postprocess
        self.objecttive = None

        self.params = []
        self.iterates = []
        self.steps = []
        self.constraint_sets = []

        # TODO: think if we need to separate out param/iterate bounds
        # self.param_lower_bounds = {}
        # self.param_upper_bounds = {}
        # self.iterate_lower_bounds = {}
        # self.iterate_upper_bounds = {}

        if solver == 'gurobi':
            self.canonicalizer = GurobiCanonicalizer()

        self.lower_bounds = {}
        self.upper_bounds = {}

    def add_param(self, n, lb=-np.inf, ub=np.inf):
        assert np.all(lb <= ub)
        param = Vector(n)
        self.params.append(param)
        self.lower_bounds[param] = upcast(n, lb)
        self.upper_bounds[param] = upcast(n, ub)
        self.canonicalizer.add_param_var(param, lb=lb, ub=ub)

        return param

    def add_initial_iterate(self, n, lb=-np.inf, ub=np.inf):
        assert np.all(lb <= ub)
        iterate = Vector(n)
        self.iterates.append(iterate)
        self.lower_bounds[iterate] = upcast(n, lb)
        self.upper_bounds[iterate] = upcast(n, ub)
        self.canonicalizer.add_initial_iterate_var(iterate, lb=lb, ub=ub)

        return iterate

    def add_iterate(self, iterate):
        self.iterates.append(iterate)

    def add_step(self, step):
        self.steps.append(step)

    # def add_explicit_affine_step(self, rhs_expr):
    #     '''
    #         Process:
    #             - Construct out iterate
    #             - Construct affine step object
    #             - Do the interval bound propagations as necessary
    #             - Add the iterate and constraints to the canonicalizer
    #             - Do the OBBT (if self.num_obbt >= 0`)
    #             - Return the out iterate Vector object
    #     '''
    #     out_iterate = Vector(rhs_expr.get_output_dim())
    #     step = AffineStep(out_iterate, rhs_expr)

    #     # TODO: Note that this might not be necessary at all given the LinExpr structure

    #     return out_iterate

    def relu_step(self, rhs_expr):
        out_iterate = Vector(rhs_expr.get_output_dim())
        step = ReluStep(out_iterate, rhs_expr)

        rhs_lb, rhs_ub = self.linear_bound_prop(rhs_expr)
        # step.rhs_lb = rhs_lb  # need to update lb/ub
        # step.rhs_ub = rhs_ub
        step.update_rhs_lb(rhs_lb)
        step.update_rhs_ub(rhs_ub)

        out_lb = relu(rhs_lb)
        out_ub = relu(rhs_ub)

        self.iterates.append(out_iterate)
        self.lower_bounds[out_iterate] = out_lb
        self.upper_bounds[out_iterate] = out_ub

        self.canonicalizer.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
        # TODO: add constraints to the gurobi problem
        self.canonicalizer.add_relu_constraints(step)

        # TODO: OBBT

        return out_iterate

    def add_constraint_set(self, constraint_set):
        self.constraint_sets.append(constraint_set)

    def linear_bound_prop(self, expr):
        n = expr.get_output_dim()
        out_lb = np.zeros(n)
        out_ub = np.zeros(n)

        for key, value in expr.decomposition_dict.items():
            x_lb = self.lower_bounds[key]
            x_ub = self.upper_bounds[key]
            Ax_lower, Ax_upper = interval_bound_prop(value, x_lb, x_ub)
            assert np.all(Ax_lower <= Ax_upper)

            out_lb += Ax_lower
            out_ub += Ax_upper

        assert np.all(out_lb <= out_ub)

        return out_lb, out_ub

    def set_zero_objective(self):
        # self.objective = obj

        # TODO: replace once we have the infty norm objectives
        self.canonicalizer.set_zero_objective()

    def set_infinity_norm_objective(self, expr):
        expr_lb, expr_ub = self.linear_bound_prop(expr)
        self.canonicalizer.set_infinity_norm_objective(expr, expr_lb, expr_ub)

    def solve(self):
        self.canonicalizer.solve_model()


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3

    absA = np.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_lower, Ax_upper


def upcast(n, val):
    '''
        Utility function that returns val * np.ones(n) if it is not already a vector.
    '''
    if isinstance(val, int) or isinstance(val, float):
        return float(val) * np.ones(n)
    elif isinstance(val, np.ndarray):
        return val
    else:
        raise TypeError(f'lb or ub needs to be an int, float, or np.adarray. Got {type(val)}')


def relu(v):
    return np.maximum(v, 0)

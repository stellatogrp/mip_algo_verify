import numpy as np

from mipalgover.canonicalizers.gurobi_canonicalizer import GurobiCanonicalizer
from mipalgover.vector import Vector


class Verifier(object):

    def __init__(self,
                 num_obbt=3,
                 postprocess=False,
                 solver='gurobi'):

        self.num_obbt = num_obbt
        self.postprocess = postprocess

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

    def add_explicit_linear_step(self, rhs_expr):
        '''
            Process:
                - Construct out iterate and add it to the canonicalizer
                - Construct affine step object
                - Do the bound propagations as necessary
                - Return the out iterate Vector object
        '''
        out_iterate = Vector(rhs_expr.get_output_dim())
        # step = AffineStep(out_iterate, rhs_expr)

        # TODO: finish

        return out_iterate

    def add_constraint_set(self, constraint_set):
        self.constraint_sets.append(constraint_set)


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

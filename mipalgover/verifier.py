import numpy as np

from mipalgover.canonicalizers.gurobi_canonicalizer import GurobiCanonicalizer


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

    def add_param(self, param, lb, ub):
        assert np.all(lb <= ub)
        self.params.append(param)
        self.lower_bounds[param] = lb
        self.upper_bounds[param] = ub
        self.canonicalizer.add_param_var(param, lb=lb, ub=ub)

    def add_initial_iterate(self, iterate, lb, ub):
        self.iterates.append(iterate)

    def add_iterate(self, iterate):
        self.iterates.append(iterate)

    def add_step(self, step):
        self.steps.append(step)

    def add_constraint_set(self, constraint_set):
        self.constraint_sets.append(constraint_set)


def interval_bound_prop(A, l, u):
    # given x in [l, u], give bounds on Ax
    # using techniques from arXiv:1810.12715, Sec. 3
    absA = np.abs(A)
    Ax_upper = .5 * (A @ (u + l) + absA @ (u - l))
    Ax_lower = .5 * (A @ (u + l) - absA @ (u - l))
    return Ax_lower, Ax_upper

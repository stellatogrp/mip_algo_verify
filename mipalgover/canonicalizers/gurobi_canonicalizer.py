import gurobipy as gp
import numpy as np


class GurobiCanonicalizer(object):

    def __init__(self,
                 gurobi_params={}):
        self.vector_var_map = {}
        self.model = None
        self.model = gp.Model()
        # TODO: initialize model with gurobi params

    def add_param_var(self, param, lb=-np.inf, ub=np.inf):
        self.vector_var_map[param] = self.model.addMVar(param.n, lb=lb, ub=ub)

    def add_initial_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self.vector_var_map[iterate] = self.model.addMVar(iterate.n, lb=lb, ub=ub)

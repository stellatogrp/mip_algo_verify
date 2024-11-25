import gurobipy as gp
import numpy as np


class GurobiCanonicalizer(object):

    def __init__(self,
                 gurobi_params={}):
        self.vector_var_map = {}
        self.model = None
        self.model = gp.Model()
        self.model_to_opt = None
        # TODO: initialize model with gurobi params
        self.step_constr_map = {}
        self.obj_constraints = []

    def add_param_var(self, param, lb=-np.inf, ub=np.inf):
        self.vector_var_map[param] = self.model.addMVar(param.n, lb=lb, ub=ub)

    def add_initial_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self.vector_var_map[iterate] = self.model.addMVar(iterate.n, lb=lb, ub=ub)

    def add_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self.vector_var_map[iterate] = self.model.addMVar(iterate.n, lb=lb, ub=ub)

    def update_vector_bounds(self, iterate, lb, ub):
        gp_var = self.vector_var_map[iterate]

        old_lb = gp_var.lb
        gp_var.lb = np.maximum(old_lb, lb)

        old_ub = gp_var.ub
        gp_var.ub = np.minimum(old_ub, ub)

    def lin_expr_to_gp_expr(self, linexpr):
        gp_expr = None
        for key, value in linexpr.decomposition_dict.items():
            gp_var = self.vector_var_map[key]

            if gp_expr is None:
                gp_expr = value @ gp_var
            else:
                gp_expr += value @ gp_var

        return gp_expr

    def add_relu_constraints(self, step):
        lhs = step.lhs_expr
        rhs = step.rhs_expr
        assert lhs in self.vector_var_map

        lhs_gp_expr = self.lin_expr_to_gp_expr(lhs)
        rhs_gp_expr = self.lin_expr_to_gp_expr(rhs)
        rhs_lb = step.rhs_lb
        rhs_ub = step.rhs_ub
        n = lhs.get_output_dim()

        out_constraints = []
        out_new_vars = {}

        for i in range(n):
            if rhs_ub[i] <= 0:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == 0))
            elif rhs_lb[i] > 0:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i]))
            else:
                out_new_vars[i] = self.model.addVar(vtype=gp.GRB.BINARY)
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_ub[i] / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i])))
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i]))
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] - rhs_lb[i] * (1 - out_new_vars[i])))
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_ub[i] * out_new_vars[i]))

        self.step_constr_map[step] = out_constraints
        self.model.update()

    def set_zero_objective(self):
        # this function is mostly just for debugging the constraints
        self.model.setObjective(0)
        self.model.update()

    def set_infinity_norm_objective(self, expr, lb, ub):
        n = expr.get_output_dim()
        self.model_to_opt = self.model.copy()
        rhs_gp_expr = self.lin_expr_to_gp_expr(expr)
        # print(rhs_gp_expr)

        vobj = self.model_to_opt.addMVar(expr.get_output_dim(), vtype=gp.GRB.BINARY)
        up = self.model_to_opt.addMVar(expr.get_output_dim(), ub=np.abs(ub))
        un = self.model_to_opt.addMVar(expr.get_output_dim(), ub=np.abs(lb))

        self.model_to_opt.addConstr(up - un == rhs_gp_expr)

        for i in range(n):
            if lb[i] >= 0:
                # self.model_to_opt.addConstr(up[i] == rhs_gp_expr[i])
                self.model_to_opt.addConstr(un[i] == 0)
            elif ub[i] < 0:
                # self.model_to_opt.addConstr(un[i] == -rhs_gp_expr[i])
                self.model_to_opt.addConstr(up[i] == 0)
            else:
                self.model_to_opt.addConstr(up[i] <= np.abs(ub) * vobj[i])
                self.model_to_opt.addConstr(un[i] <= np.abs(lb) * (1-vobj[i]))

        M = np.maximum(np.max(np.abs(ub)), np.max(np.abs(lb)))
        q = self.model_to_opt.addVar(ub=M)
        gamma = self.model_to_opt.addMVar(n, vtype=gp.GRB.BINARY)

        for i in range(n):
            self.model_to_opt.addConstr(q >= up[i] + un[i])
            self.model_to_opt.addConstr(q <= up[i] + un[i] + M * (1 - gamma[i]))

        self.model_to_opt.setObjective(q, gp.GRB.MAXIMIZE)
        self.model_to_opt.update()

    def solve_model(self):
        self.model_to_opt.optimize()
        # self.model.optimize()

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

        self.model_to_opt = self.model.copy()
        self.model_to_opt.setObjective(0)
        self.model_to_opt.update()

    def set_infinity_norm_objective(self, expr_list, lb_list, ub_list):
        self.model_to_opt = self.model.copy()

        all_lb = np.hstack(lb_list)
        all_ub = np.hstack(ub_list)

        M = np.maximum(np.max(np.abs(all_ub)), np.max(np.abs(all_lb)))
        q = self.model_to_opt.addVar(ub=M)

        for expr, lb, ub in zip(expr_list, lb_list, ub_list):
            n = expr.get_output_dim()
            rhs_gp_expr = self.lin_expr_to_gp_expr(expr)

            up = self.model_to_opt.addMVar(n, ub=np.abs(ub))
            un = self.model_to_opt.addMVar(n, ub=np.abs(lb))

            self.model_to_opt.addConstr(up - un == rhs_gp_expr)

            for i in range(n):
                if lb[i] >= 0:
                    self.model_to_opt.addConstr(un[i] == 0)
                elif ub[i] < 0:
                    self.model_to_opt.addConstr(up[i] == 0)
                else:
                    vobji = self.model_to_opt.addMVar(1, vtype=gp.GRB.BINARY)
                    self.model_to_opt.addConstr(up[i] <= np.abs(ub[i]) * vobji)
                    self.model_to_opt.addConstr(un[i] <= np.abs(lb[i]) * (1-vobji))

            gamma = self.model_to_opt.addMVar(n, vtype=gp.GRB.BINARY)
            for i in range(n):
                self.model_to_opt.addConstr(q >= up[i] + un[i])
                self.model_to_opt.addConstr(q <= up[i] + un[i] + M * (1 - gamma[i]))

        self.model_to_opt.setObjective(q, gp.GRB.MAXIMIZE)
        self.model_to_opt.update()

    def solve_model(self):
        self.model_to_opt.optimize()
        return self.model_to_opt.objVal

    def extract_sol(self, iterate):
        out = 0
        for key, value in iterate.decomposition_dict.items():
            gp_var = self.vector_var_map[key]
            gp_var_val = np.zeros(gp_var.shape)

            for i in range(gp_var.shape[0]):
                # print(i)
                # rel_model.getVarByName(var.VarName.item()).X
                # print(self.model_to_opt.getVarByName(gp_var[i].VarName.item()).X)
                gp_var_val[i] = self.model_to_opt.getVarByName(gp_var[i].VarName.item()).X

            out += value @ gp_var_val

        return out

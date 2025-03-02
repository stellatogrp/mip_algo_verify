import logging

import gurobipy as gp
import numpy as np

log = logging.getLogger(__name__)


class GurobiOBBTHandler(object):

    def __init__(self):
        self.vector_var_map = {}
        self.model = gp.Model()

        self.model.Params.OutputFlag = 0

        # for key, val in gurobi_params.items():
        #     self.model.setParam(key, val)

        self.model_to_opt = None
        self.step_constr_map = {}

    def add_param_var(self, param, lb=-np.inf, ub=np.inf):
        self.vector_var_map[param] = self.model.addMVar(param.n, lb=lb, ub=ub)

    def add_initial_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self.vector_var_map[iterate] = self.model.addMVar(iterate.n, lb=lb, ub=ub)

    def add_iterate_var(self, iterate, lb=-np.inf, ub=np.inf):
        self.vector_var_map[iterate] = self.model.addMVar(iterate.n, lb=lb, ub=ub)

    def add_equality_constraint(self, lhs_expr, rhs_expr):
        lhs_gp_expr = self.lin_expr_to_gp_expr(lhs_expr)
        rhs_gp_expr = self.lin_expr_to_gp_expr(rhs_expr)

        # out_constraints = []
        # out_constraints.append(self.model.addConstr(lhs_gp_expr == rhs_gp_expr))

        self.model.addConstr(lhs_gp_expr == rhs_gp_expr)
        self.model.update()

    def update_vector_bounds(self, iterate, lb, ub):
        gp_var = self.vector_var_map[iterate]

        old_lb = gp_var.lb
        gp_var.lb = np.maximum(old_lb, lb)

        old_ub = gp_var.ub
        gp_var.ub = np.minimum(old_ub, ub)

    def leaf_expr_to_gp_var(self, leaf):
        assert leaf.is_leaf

        key = next(iter(leaf.decomposition_dict))
        return self.vector_var_map[key]

    def lin_expr_to_gp_expr(self, linexpr):
        gp_expr = None
        for key, value in linexpr.decomposition_dict.items():
            gp_var = self.vector_var_map[key]

            if gp_expr is None:
                gp_expr = value @ gp_var
            else:
                gp_expr += value @ gp_var

        return gp_expr

    def obbt(self, linexpr):
        self.model.update()
        n = linexpr.get_output_dim()

        lb_out = np.zeros(n)
        ub_out = np.zeros(n)

        gp_expr = self.lin_expr_to_gp_expr(linexpr)

        for sense in [gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE]:
            for i in range(n):
                self.model.setObjective(gp_expr[i], sense)
                self.model.update()

                self.model.update()
                self.model.optimize()

                if self.model.status != gp.GRB.OPTIMAL:
                    log.info(f'bound tighting failed, GRB model status: {self.model.status}')
                    log.info(f'i = {i}, sense={sense}')
                    exit(0)
                    return None

                if sense == gp.GRB.MAXIMIZE:
                    ub_out[i] = self.model.objVal
                else:
                    lb_out[i] = self.model.objVal

        # reset objective
        self.model.setObjective(0)

        assert np.all(lb_out <= ub_out)

        return lb_out, ub_out

    def add_relu_constraints(self, step, lhs_lb, lhs_ub):
        lhs = step.lhs_expr
        rhs = step.rhs_expr
        assert lhs in self.vector_var_map

        lhs_gp_expr = self.lin_expr_to_gp_expr(lhs)
        rhs_gp_expr = self.lin_expr_to_gp_expr(rhs)
        rhs_lb = step.rhs_lb
        rhs_ub = step.rhs_ub
        # n = lhs.get_output_dim()

        out_constraints = []

        proj_indices = step.proj_indices
        nonproj_indices = step.nonproj_indices

        # after ranges are added, remove the range(n) and add the constraints when needed
        # make sure to enforce equality constraints between lhs and rhs for other indices

        for i in nonproj_indices:
            out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i]))

        for i in proj_indices:
            if rhs_ub[i] <= 0:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == 0))
            elif rhs_lb[i] > 0:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i]))
            else:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= 0))
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_ub[i] / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i])))
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i]))

        self.step_constr_map[step] = out_constraints
        self.model.update()


    def add_saturated_linear_constraints(self, step, lhs_lb, lhs_ub):
        lhs = step.lhs_expr
        rhs = step.rhs_expr
        assert lhs in self.vector_var_map
        l, u = step.l, step.u

        lhs_gp_expr = self.lin_expr_to_gp_expr(lhs)
        rhs_gp_expr = self.lin_expr_to_gp_expr(rhs)
        rhs_lb = step.rhs_lb
        rhs_ub = step.rhs_ub
        n = lhs.get_output_dim()

        out_constraints = []

        for i in range(n):
            if rhs_lb[i] >= u[i]:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == u[i]))
            elif rhs_ub[i] <= l[i]:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == l[i]))
            elif rhs_lb[i] >= l[i] and rhs_ub[i] <= u[i]:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i]))
            else:
                if rhs_lb[i] < l[i] and rhs_ub[i] > u[i]:
                    # following two are redundant due to preprocessing bounds on the left hand side iterate
                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= u[i]))
                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= l[i]))

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= (u[i] - l[i]) / (u[i] - rhs_lb[i]) * (rhs_gp_expr[i] - u[i]) + u[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= (u[i] - l[i]) / (rhs_ub[i] - l[i]) * (rhs_gp_expr[i] - l[i]) + l[i]))

                elif rhs_lb[i] < l[i] and rhs_ub[i] < u[i]:
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= (rhs_ub[i] - l[i]) / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_ub[i]) + rhs_ub[i]))

                elif rhs_lb[i] > l[i] and rhs_ub[i] > u[i]:
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i]))

                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= (u[i] - l[i]) / (rhs_ub[i] - l[i]) * (rhs_gp_expr[i] - l[i]) + l[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= (u[i] - rhs_lb[i]) / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i]) + rhs_lb[i]))

                else:
                    raise RuntimeError('Unreachable code')

        self.step_constr_map[step] = out_constraints
        self.model.update()


    def add_soft_threshold_constraints(self, step, lhs_lb, lhs_ub):
        lhs = step.lhs_expr
        rhs = step.rhs_expr
        assert lhs in self.vector_var_map
        lambd = step.lambd

        lhs_gp_expr = self.lin_expr_to_gp_expr(lhs)
        rhs_gp_expr = self.lin_expr_to_gp_expr(rhs)
        rhs_lb = step.rhs_lb
        rhs_ub = step.rhs_ub
        n = lhs.get_output_dim()

        out_constraints = []

        for i in range(n):
            if rhs_lb[i] >= lambd:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i] - lambd))
            elif rhs_ub[i] <= -lambd:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i] + lambd))
            elif rhs_lb[i] >= -lambd and rhs_ub[i] <= lambd:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == 0.0))
            else:
                if rhs_lb[i] < -lambd and rhs_ub[i] > lambd:
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i] - lambd))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] + lambd))

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= lhs_ub[i] / (rhs_ub[i] + lambd) * (rhs_gp_expr[i] + lambd)))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= lhs_lb[i] / (rhs_lb[i] - lambd) * (rhs_gp_expr[i] - lambd)))

                elif -lambd <= rhs_lb[i] <= lambd and rhs_ub[i] > lambd:
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= 0))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= lhs_ub[i] / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i])))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i] - lambd))

                elif -lambd <= rhs_ub[i] <= lambd and rhs_lb[i] <= -lambd:

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= 0))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= lhs_lb[i] / (rhs_lb[i] - rhs_ub[i]) * (rhs_gp_expr[i] - rhs_ub[i])))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] + lambd))

                else:
                    raise RuntimeError('Unreachable code')

        self.step_constr_map[step] = out_constraints
        self.model.update()

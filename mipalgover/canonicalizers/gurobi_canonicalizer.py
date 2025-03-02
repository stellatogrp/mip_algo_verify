import logging

import gurobipy as gp
import numpy as np
import scipy.sparse as spa

from mipalgover.canonicalizers.huchette_cuts.relu import relu_add_conv_cuts, relu_create_new_constr
from mipalgover.canonicalizers.huchette_cuts.soft_threshold import (
    st_add_neg_conv_cuts,
    st_add_pos_conv_cuts,
    st_create_new_neg_constr,
    st_create_new_pos_constr,
)
from mipalgover.steps.relu import ReluStep
from mipalgover.steps.soft_threshold import SoftThresholdStep

log = logging.getLogger(__name__)


class GurobiCanonicalizer(object):

    def __init__(self,
                 gurobi_params={}):
        self.vector_var_map = {}
        self.model = gp.Model()

        for key, val in gurobi_params.items():
            self.model.setParam(key, val)

        self.model_to_opt = None
        self.step_constr_map = {}
        self.obj_constraints = []

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

    def add_relu_constraints(self, step):
        lhs = step.lhs_expr
        rhs = step.rhs_expr
        assert lhs in self.vector_var_map

        lhs_gp_expr = self.lin_expr_to_gp_expr(lhs)
        rhs_gp_expr = self.lin_expr_to_gp_expr(rhs)
        rhs_lb = step.rhs_lb
        rhs_ub = step.rhs_ub
        # n = lhs.get_output_dim()

        out_constraints = []
        out_new_vars = {}

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
                if step.relax_binary_vars:
                    out_new_vars[i] = self.model.addVar(lb=0., ub=1.)
                else:
                    out_new_vars[i] = self.model.addVar(vtype=gp.GRB.BINARY)
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_ub[i] / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i])))
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i]))
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] - rhs_lb[i] * (1 - out_new_vars[i])))
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_ub[i] * out_new_vars[i]))

                step.idx_with_binary_vars.add(i)

        self.step_constr_map[step] = out_constraints
        self.model.update()

    def add_saturated_linear_constraints(self, step, out_lb, out_ub):
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
        new_w1 = {}
        new_w2 = {}

        for i in range(n):
            if rhs_lb[i] >= u[i]:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == u[i]))
            elif rhs_ub[i] <= l[i]:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == l[i]))
            elif rhs_lb[i] >= l[i] and rhs_ub[i] <= u[i]:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i]))
            else:
                if rhs_lb[i] < l[i] and rhs_ub[i] > u[i]:
                    if step.relax_binary_vars:
                        new_w1[i] = self.model.addVar(lb=0., ub=1.)
                        new_w2[i] = self.model.addVar(lb=0., ub=1.)
                    else:
                        new_w1[i] = self.model.addVar(vtype=gp.GRB.BINARY)
                        new_w2[i] = self.model.addVar(vtype=gp.GRB.BINARY)

                    # following two are redundant due to preprocessing bounds on the left hand side iterate
                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= u[i]))
                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= l[i]))

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= (u[i] - l[i]) / (u[i] - rhs_lb[i]) * (rhs_gp_expr[i] - u[i]) + u[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= (u[i] - l[i]) / (rhs_ub[i] - l[i]) * (rhs_gp_expr[i] - l[i]) + l[i]))

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= u[i] - (u[i] - l[i]) * (1 - new_w1[i])))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= l[i] + (u[i] - l[i]) * (1 - new_w2[i])))

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i] - (rhs_ub[i] - rhs_lb[i]) * new_w1[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] + (rhs_ub[i] - rhs_lb[i]) * new_w2[i]))

                    out_constraints.append(self.model.addConstr(rhs_gp_expr[i] >= u[i] + (rhs_lb[i] - u[i]) * (1 - new_w1[i])))
                    out_constraints.append(self.model.addConstr(rhs_gp_expr[i] <= u[i] + (rhs_ub[i] - u[i]) * new_w1[i]))

                    out_constraints.append(self.model.addConstr(rhs_gp_expr[i] >= l[i] + (rhs_lb[i] - l[i]) * new_w2[i]))
                    out_constraints.append(self.model.addConstr(rhs_gp_expr[i] <= l[i] + (rhs_ub[i] - l[i]) * (1 - new_w2[i])))

                    out_constraints.append(self.model.addConstr(new_w1[i] + new_w2[i] <= 1))
                elif rhs_lb[i] < l[i] and rhs_ub[i] < u[i]:
                    if step.relax_binary_vars:
                        new_w2[i] = self.model.addVar(lb=0., ub=1.)
                    else:
                        new_w2[i] = self.model.addVar(vtype=gp.GRB.BINARY)
                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= l[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i]))

                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= (u[i] - l[i]) / (u[i] - rhs_lb[i]) * (rhs_gp_expr[i] - u[i]) + u[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= (rhs_ub[i] - l[i]) / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_ub[i]) + rhs_ub[i]))

                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= l[i] + (u[i] - l[i]) * (1 - new_w2[i])))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= l[i] + (rhs_ub[i] - l[i]) * (1 - new_w2[i])))

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] + (rhs_ub[i] - rhs_lb[i]) * new_w2[i]))

                    out_constraints.append(self.model.addConstr(rhs_gp_expr[i] >= l[i] + (rhs_lb[i] - l[i]) * new_w2[i]))
                    out_constraints.append(self.model.addConstr(rhs_gp_expr[i] <= l[i] + (rhs_ub[i] - l[i]) * (1 - new_w2[i])))

                elif rhs_lb[i] > l[i] and rhs_ub[i] > u[i]:
                    if step.relax_binary_vars:
                        new_w1[i] = self.model.addVar(lb=0., ub=1.)
                    else:
                        new_w1[i] = self.model.addVar(vtype=gp.GRB.BINARY)

                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= u[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i]))

                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= (u[i] - l[i]) / (rhs_ub[i] - l[i]) * (rhs_gp_expr[i] - l[i]) + l[i]))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= (u[i] - rhs_lb[i]) / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i]) + rhs_lb[i]))

                    # out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= u[i] - (u[i] - l[i]) * (1 - new_w1[i])))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= u[i] - (u[i] - rhs_lb[i]) * (1 - new_w1[i])))

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i] - (rhs_ub[i] - rhs_lb[i]) * new_w1[i]))

                    out_constraints.append(self.model.addConstr(rhs_gp_expr[i] >= u[i] + (rhs_lb[i] - u[i]) * (1 - new_w1[i])))
                    out_constraints.append(self.model.addConstr(rhs_gp_expr[i] <= u[i] + (rhs_ub[i] - u[i]) * new_w1[i]))

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
        new_w1 = {}
        new_w2 = {}

        for i in range(n):
            if rhs_lb[i] >= lambd:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i] - lambd))
            elif rhs_ub[i] <= -lambd:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == rhs_gp_expr[i] + lambd))
            elif rhs_lb[i] >= -lambd and rhs_ub[i] <= lambd:
                out_constraints.append(self.model.addConstr(lhs_gp_expr[i] == 0.0))
            else:
                if rhs_lb[i] < -lambd and rhs_ub[i] > lambd:
                    # log.info(f'double side: {i}')
                    # if step.relax_binary_vars:
                    #     new_w1[i] = self.model.addVar(lb=0., ub=1.)
                    #     new_w2[i] = self.model.addVar(lb=0., ub=1.)
                    # else:
                    if not step.relax_binary_vars:
                        new_w1[i] = self.model.addVar(vtype=gp.GRB.BINARY)
                        new_w2[i] = self.model.addVar(vtype=gp.GRB.BINARY)
                        step.idx_with_right_binary_vars.add(i)
                        step.idx_with_left_binary_vars.add(i)

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i] - lambd))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] + lambd))

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= lhs_ub[i] / (rhs_ub[i] + lambd) * (rhs_gp_expr[i] + lambd)))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= lhs_lb[i] / (rhs_lb[i] - lambd) * (rhs_gp_expr[i] - lambd)))

                    if not step.relax_binary_vars:
                        # Upper right part: w1 = 1, y >= lambda_t
                        out_constraints.append(self.model.addConstr(rhs_gp_expr[i] >= lambd + (rhs_lb[i] - lambd) * (1-new_w1[i])))
                        out_constraints.append(self.model.addConstr(rhs_gp_expr[i] <= lambd + (rhs_ub[i] - lambd) * new_w1[i]))

                        # Lower left part: w2 = 1, y <= -lambda_t
                        out_constraints.append(self.model.addConstr(rhs_gp_expr[i] <= -lambd + (rhs_ub[i] + lambd) * (1-new_w2[i])))
                        out_constraints.append(self.model.addConstr(rhs_gp_expr[i] >= -lambd + (rhs_lb[i] + lambd) * new_w2[i]))

                        out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] - lambd + (2 * lambd) * (1-new_w1[i])))
                        out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i] + lambd + (-2 * lambd) * (1-new_w2[i])))

                        out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= lhs_ub[i] * new_w1[i]))
                        out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= lhs_lb[i] * new_w2[i]))

                        out_constraints.append(self.model.addConstr(new_w1[i] + new_w2[i] <= 1))

                elif -lambd <= rhs_lb[i] <= lambd and rhs_ub[i] > lambd:
                    # log.info(f'right side only: {i}')
                    # if step.relax_binary_vars:
                    #     new_w1[i] = self.model.addVar(lb=0., ub=1.)
                    # else:
                    if not step.relax_binary_vars:
                        new_w1[i] = self.model.addVar(vtype=gp.GRB.BINARY)
                        step.idx_with_right_binary_vars.add(i)

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= 0))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= lhs_ub[i] / (rhs_ub[i] - rhs_lb[i]) * (rhs_gp_expr[i] - rhs_lb[i])))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i] - lambd))

                    if not step.relax_binary_vars:
                    # Upper right part: w1 = 1, y >= lambda_t
                        out_constraints.append(self.model.addConstr(rhs_gp_expr[i] >= lambd + (rhs_lb[i] - lambd) * (1 - new_w1[i])))
                        out_constraints.append(self.model.addConstr(rhs_gp_expr[i] <= lambd + (rhs_ub[i] - lambd) * new_w1[i]))

                        out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] - lambd + (2 * lambd)*(1 - new_w1[i])))
                        out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= lhs_ub[i] * new_w1[i]))

                elif -lambd <= rhs_ub[i] <= lambd and rhs_lb[i] <= -lambd:
                    # log.info(f'left side only: {i}')
                    # if step.relax_binary_vars:
                    #     new_w2[i] = self.model.addVar(lb=0., ub=1.)
                    # else:
                    if not step.relax_binary_vars:
                        new_w2[i] = self.model.addVar(vtype=gp.GRB.BINARY)
                        step.idx_with_left_binary_vars.add(i)

                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= 0))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= lhs_lb[i] / (rhs_lb[i] - rhs_ub[i]) * (rhs_gp_expr[i] - rhs_ub[i])))
                    out_constraints.append(self.model.addConstr(lhs_gp_expr[i] <= rhs_gp_expr[i] + lambd))

                    if not step.relax_binary_vars:
                        # Lower left part: w2 = 1, y <= -lambda_t
                        out_constraints.append(self.model.addConstr(rhs_gp_expr[i] <= -lambd + (rhs_ub[i] + lambd) * (1 - new_w2[i])))
                        out_constraints.append(self.model.addConstr(rhs_gp_expr[i] >= -lambd + (rhs_lb[i] + lambd) * new_w2[i]))

                        out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= rhs_gp_expr[i] + lambd + (-2 * lambd)*(1 - new_w2[i])))
                        out_constraints.append(self.model.addConstr(lhs_gp_expr[i] >= lhs_lb[i] * new_w2[i]))

                else:
                    raise RuntimeError('Unreachable code')

        self.step_constr_map[step] = out_constraints
        self.model.update()

    def obbt(self, linexpr):
        # create/solve the obbt problem for all elements in linexpr
        self.model.update()
        n = linexpr.get_output_dim()

        lb_out = np.zeros(n)
        ub_out = np.zeros(n)

        gp_expr = self.lin_expr_to_gp_expr(linexpr)
        # log.info('trying outside of loop')
        # self.model.setObjective(gp_expr[39], 1)
        # self.model.optimize()

        # log.info(f'full model status: {self.model.status}')
        # obbt_model = self.model.relax()
        # obbt_model.Params.OutputFlag = 0
        # obbt_model.update()
        # obbt_model.optimize()
        # log.info(f'obbt model status: {obbt_model.status}')
        # if obbt_model.status == 3:
        #     log.info('computing IIS')
        #     obbt_model.computeIIS()


        for sense in [gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE]:
            for i in range(n):
                self.model.setObjective(gp_expr[i], sense)
                self.model.update()

                obbt_model = self.model.relax()
                obbt_model.Params.OutputFlag = 0
                obbt_model.update()
                obbt_model.optimize()

                if obbt_model.status != gp.GRB.OPTIMAL:
                    log.info(f'bound tighting failed, GRB model status: {obbt_model.status}')
                    log.info(f'i = {i}, sense={sense}')
                    exit(0)
                    return None

                if sense == gp.GRB.MAXIMIZE:
                    ub_out[i] = obbt_model.objVal
                else:
                    lb_out[i] = obbt_model.objVal

        # reset objective
        self.model.setObjective(0)

        assert np.all(lb_out <= ub_out)

        return lb_out, ub_out

    def add_theory_cut(self, C, target_expr, bound_lb, bound_ub):
        if C == np.inf:
            return
        target_gp_expr = self.lin_expr_to_gp_expr(target_expr)
        if target_expr.is_leaf:
            # directly update the lb/ub parameters if possible
            target_gp_expr.lb = bound_lb - C
            target_gp_expr.ub = bound_ub + C
        else:
            self.model.addConstr(target_gp_expr >= bound_lb - C)
            self.model.addConstr(target_gp_expr <= bound_ub + C)
        self.model.update()

    def post_process(self, target_expr, bound_lb, bound_ub):
        target_gp_expr = self.lin_expr_to_gp_expr(target_expr)
        if target_expr.is_leaf:
            # directly update the lb/ub parameters if possible
            target_gp_expr.lb = bound_lb
            target_gp_expr.ub = bound_ub
        else:
            self.model.addConstr(target_gp_expr >= bound_lb)
            self.model.addConstr(target_gp_expr <= bound_ub)
        self.model.update()

    def equality_constraint(self, lhs_expr, rhs_expr):
        lhs_gp_expr = self.lin_expr_to_gp_expr(lhs_expr)
        rhs_gp_expr = self.lin_expr_to_gp_expr(rhs_expr)
        self.model.addConstr(lhs_gp_expr == rhs_gp_expr)
        self.model.update()

    def set_zero_objective(self):
        # this function is mostly just for debugging the constraints

        self.model_to_opt = self.model.copy()
        self.model_to_opt.setObjective(0)
        self.model_to_opt.update()

    def set_infinity_norm_objective(self, expr_list, lb_list, ub_list):
        self.model.update()
        if self.model_to_opt is not None:
            self.model_to_opt.dispose()
        self.model_to_opt = self.model.copy()

        all_lb = np.hstack(lb_list)
        all_ub = np.hstack(ub_list)

        M = np.maximum(np.max(np.abs(all_ub)), np.max(np.abs(all_lb)))
        q = self.model_to_opt.addVar(ub=M)

        all_gammas = []

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

            all_gammas.append(gamma)

        # need 1 gamma across all vectors in the list
        gamma_constr = 0
        for gamma in all_gammas:
            gamma_constr += gp.quicksum(gamma)
        self.model_to_opt.addConstr(gamma_constr == 1)

        self.model_to_opt.setObjective(q, gp.GRB.MAXIMIZE)
        self.model_to_opt.update()

    def solve_model(self, steps, lower_bounds, upper_bounds, **kwargs):
        if 'huchette_cuts' in kwargs:
            if kwargs['huchette_cuts']:
                self.add_huchette_cuts(steps, lower_bounds, upper_bounds, **kwargs)

        if 'include_rel_LP_sol' in kwargs:
            if kwargs['include_rel_LP_sol']:
                rel_model = self.model_to_opt.relax()
                rel_model.optimize()
                self.rel_LP_sol = rel_model.objVal
                # print('relaxed LP sol:', self.rel_LP_sol)
            else:
                self.rel_LP_sol = None
        else:
            self.rel_LP_sol = None
        self.model_to_opt.optimize()
        return self.model_to_opt.objVal

    def add_huchette_cuts(self, steps, lower_bounds, upper_bounds, **kwargs):
        print('adding huchette cuts')
        # print(kwargs)
        # print(steps)
        relaxed_model = self.model_to_opt.relax()
        relaxed_model.optimize()
        print('relaxed model obj:', relaxed_model.objVal)

        for step in steps:
            if isinstance(step, ReluStep):
                self.relu_huchette_cuts(relaxed_model, step, lower_bounds, upper_bounds, **kwargs)
            elif isinstance(step, SoftThresholdStep):
                self.soft_threshold_huchette_cuts(relaxed_model, step, lower_bounds, upper_bounds, **kwargs)

        self.model_to_opt.update()

    def relu_huchette_cuts(self, relaxed_model, step, lower_bounds, upper_bounds, **kwargs):
        n = step.lhs_expr.get_output_dim()
        # huchette cuts for w = relu(a^T y)

        w, A, y, l, u, w_var, y_var = self.extract_vecs_for_huchette_cuts(relaxed_model, step, lower_bounds, upper_bounds, **kwargs)

        for i in range(n):
            if i in step.nonproj_indices:
                continue
            if i not in step.idx_with_binary_vars:
                continue
            Iint, lI, h, Lhat, Uhat = relu_add_conv_cuts(w[i], A[i, :], y, l, u)
            if Iint is not None:
                # print(Iint, lI, h)
                new_constr = relu_create_new_constr(w_var[i], A[i, :], y_var, Iint, lI, h, Lhat, Uhat)
                self.model_to_opt.addConstr(new_constr)

    def soft_threshold_huchette_cuts(self, relaxed_model, step, lower_bounds, upper_bounds, **kwargs):
        n = step.lhs_expr.get_output_dim()
        lambd = step.lambd
        # huchette cuts for w = soft_thresh(a^T y, lambd)

        w, A, y, l, u, w_var, y_var = self.extract_vecs_for_huchette_cuts(relaxed_model, step, lower_bounds, upper_bounds, **kwargs)

        for i in range(n):
            if i in step.idx_with_right_binary_vars and i not in step.idx_with_left_binary_vars:
                Iint, lI, h, Lhat, Uhat = st_add_pos_conv_cuts(w[i], A[i, :], y, lambd, l, u)
                if Iint is not None:
                    new_constr = st_create_new_pos_constr(w_var[i], A[i, :], y_var, lambd, Iint, lI, h, Lhat, Uhat)
                    self.model_to_opt.addConstr(new_constr)

            if i in step.idx_with_left_binary_vars and i not in step.idx_with_right_binary_vars:
                Iint, uI, h, Lhat, Uhat = st_add_neg_conv_cuts(w[i], A[i, :], y, lambd, l, u)
                if Iint is not None:
                    new_constr = st_create_new_neg_constr(w_var[i], A[i, :], y_var, lambd, Iint, uI, h, Lhat, Uhat)
                    self.model_to_opt.addConstr(new_constr)

            if i in step.idx_with_right_binary_vars and i in step.idx_with_left_binary_vars:
                if w[i] > 0:
                    Iint, lI, h, Lhat, Uhat = st_add_pos_conv_cuts(w[i], A[i, :], y, lambd, l, u)
                    if Iint is not None:
                        new_constr = st_create_new_pos_constr(w_var[i], A[i, :], y_var, lambd, Iint, lI, h, Lhat, Uhat)
                        self.model_to_opt.addConstr(new_constr)
                if w[i] < 0:
                    Iint, uI, h, Lhat, Uhat = st_add_neg_conv_cuts(w[i], A[i, :], y, lambd, l, u)
                    if Iint is not None:
                        new_constr = st_create_new_neg_constr(w_var[i], A[i, :], y_var, lambd, Iint, uI, h, Lhat, Uhat)
                        self.model_to_opt.addConstr(new_constr)

    def extract_vecs_for_huchette_cuts(self, relaxed_model, step, lower_bounds, upper_bounds, **kwargs):
        # n = step.lhs_expr.get_output_dim()
        # huchette cuts for w = relu(a^T y)

        w_var = self.leaf_expr_to_gp_var(step.lhs_expr)
        w_var = self.extract_vec_var_by_name(w_var, relaxed_model)
        w = w_var.X

        rhs = step.rhs_expr

        A = []
        y_var = []
        l = []
        u = []
        for key, val in rhs.decomposition_dict.items():
            # print(key, val)
            if spa.issparse(val):
                A.append(val.todense())
            else:
                A.append(val)
            yi = self.leaf_expr_to_gp_var(key)
            y_var.append(self.extract_vec_var_by_name(yi, relaxed_model))
            l.append(lower_bounds[key])
            u.append(upper_bounds[key])
        A = np.asarray(np.hstack(A))
        y_var = gp.hstack(y_var)
        y = gp.hstack(y_var).X
        l = np.hstack(l)
        u = np.hstack(u)

        return w, A, y, l, u, w_var, y_var

    def extract_sol(self, iterate):
        out = 0
        for key, value in iterate.decomposition_dict.items():
            gp_var = self.vector_var_map[key]
            gp_var_val = np.zeros(gp_var.shape)

            for i in range(gp_var.shape[0]):
                gp_var_val[i] = self.model_to_opt.getVarByName(gp_var[i].VarName.item()).X

            out += value @ gp_var_val

        return out

    def extract_var_by_name(self, var, new_model):
        return new_model.getVarByName(var.VarName)

    def extract_vec_var_by_name(self, var, new_model):
        out = []
        for i in range(var.shape[0]):
            # print(new_model.getVarByName(var[i].item().VarName))
            out.append(new_model.getVarByName(var[i].item().VarName))
        return gp.MVar.fromlist(out)

    def get_solve_data(self):
        model = self.model_to_opt
        return {
            'objVal': model.objVal,
            'objBound': model.objBound,
            'MIPGap': model.MIPGap,
            'Runtime': model.Runtime,
            'rel_LP_sol': self.rel_LP_sol,
            'numBinVars': model.NumBinVars,
        }

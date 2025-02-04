import numpy as np

from mipalgover.canonicalizers.gurobi_canonicalizer import GurobiCanonicalizer
from mipalgover.steps.affine import AffineStep
from mipalgover.steps.relu import ReluStep
from mipalgover.steps.saturated_linear import SaturatedLinearStep
from mipalgover.steps.soft_threshold import SoftThresholdStep
from mipalgover.vector import Vector


class Verifier(object):

    def __init__(self,
                 num_obbt=3,
                 postprocess=False,
                 theory_func=None,
                 solver='gurobi',
                 solver_params={}):

        self.num_obbt = num_obbt
        self.postprocess = postprocess
        self.objective = None
        self.theory_func = theory_func
        self.obbt = True

        self.params = []
        self.iterates = []
        self.steps = []
        self.constraint_sets = []

        if solver == 'gurobi':
            self.canonicalizer = GurobiCanonicalizer(gurobi_params=solver_params)

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

    # def add_iterate(self, n):
    #     return iterate

    def add_step(self, step):
        self.steps.append(step)

    def implicit_linear_step(self, lhs_mat, rhs_expr, lhs_mat_factorization=None):
        # need to think about best way to bound prop here/set up the api
        '''
            Process:
                - Construct out iterate
                - Construct affine step object
                - Do the interval bound propagations as necessary
                - Add the iterate and constraints to the canonicalizer
                - Do the OBBT (if self.num_obbt >= 0`)
                - Return the out iterate Vector object
        '''
        iterate = Vector(lhs_mat.shape[0])
        self.iterates.append(iterate)

        lhs_expr = lhs_mat @ iterate
        step = AffineStep(lhs_expr, rhs_expr, lhs_mat=lhs_mat, lhs_mat_factorization=lhs_mat_factorization)
        Minv_rhs = step.Minv_rhs()

        iterate_lb, iterate_ub = self.linear_bound_prop(Minv_rhs)
        assert np.all(iterate_lb <= iterate_ub)

        self.lower_bounds[iterate] = iterate_lb
        self.upper_bounds[iterate] = iterate_ub

        self.canonicalizer.add_iterate_var(iterate, lb=iterate_lb, ub=iterate_ub)
        self.canonicalizer.add_equality_constraint(lhs_mat @ iterate, rhs_expr)

        self.add_step(step)

        return iterate

    def relu_step(self, rhs_expr, proj_ranges=None):
        if proj_ranges is None:
            proj_ranges = [(0, rhs_expr.get_output_dim())]
        else:
            proj_ranges = list_upcast(proj_ranges)

        out_iterate = Vector(rhs_expr.get_output_dim())
        step = ReluStep(out_iterate, rhs_expr, proj_ranges=proj_ranges)

        rhs_lb, rhs_ub = self.linear_bound_prop(rhs_expr)
        step.update_rhs_lb(rhs_lb)
        step.update_rhs_ub(rhs_ub)

        relu_rhs_lb = relu(rhs_lb, proj_ranges=proj_ranges)
        relu_rhs_ub = relu(rhs_ub, proj_ranges=proj_ranges)

        if self.obbt:
            obbt_lb, obbt_ub = self.canonicalizer.obbt(rhs_expr)

            step.update_rhs_lb(obbt_lb)
            step.update_rhs_ub(obbt_ub)

            relu_obbt_lb = relu(obbt_lb, proj_ranges=proj_ranges)
            relu_obbt_ub = relu(obbt_ub, proj_ranges=proj_ranges)

            out_lb = relu_obbt_lb
            out_ub = relu_obbt_ub
        else:
            out_lb = relu_rhs_lb
            out_ub = relu_rhs_ub

        self.iterates.append(out_iterate)
        self.lower_bounds[out_iterate] = out_lb
        self.upper_bounds[out_iterate] = out_ub

        self.canonicalizer.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
        self.canonicalizer.add_relu_constraints(step)

        self.add_step(step)

        # if convexification flag

        return out_iterate

    def saturated_linear_step(self, rhs_expr, l, u):
        out_iterate = Vector(rhs_expr.get_output_dim())
        step = SaturatedLinearStep(out_iterate, rhs_expr, l, u)

        rhs_lb, rhs_ub = self.linear_bound_prop(rhs_expr)
        step.update_rhs_lb(rhs_lb)
        step.update_rhs_ub(rhs_ub)

        sl_rhs_lb = saturated_linear(rhs_lb, l, u)
        sl_rhs_ub = saturated_linear(rhs_ub, l, u)

        if self.obbt:
            obbt_lb, obbt_ub = self.canonicalizer.obbt(rhs_expr)

            step.update_rhs_lb(obbt_lb)
            step.update_rhs_ub(obbt_ub)

            sl_obbt_lb = saturated_linear(obbt_lb, l, u)
            sl_obbt_ub = saturated_linear(obbt_ub, l, u)

            out_lb = sl_obbt_lb
            out_ub = sl_obbt_ub
        else:
            out_lb = sl_rhs_lb
            out_ub = sl_rhs_ub

        self.iterates.append(out_iterate)
        self.lower_bounds[out_iterate] = out_lb
        self.upper_bounds[out_iterate] = out_ub

        self.canonicalizer.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
        self.canonicalizer.add_saturated_linear_constraints(step, out_lb, out_ub)

        self.add_step(step)

        return out_iterate

    def soft_threshold_step(self, rhs_expr, lambd):
        out_iterate = Vector(rhs_expr.get_output_dim())
        step = SoftThresholdStep(out_iterate, rhs_expr, lambd)

        rhs_lb, rhs_ub = self.linear_bound_prop(rhs_expr)
        step.update_rhs_lb(rhs_lb)
        step.update_rhs_ub(rhs_ub)

        st_rhs_lb = soft_threshold(rhs_lb, lambd)
        st_rhs_ub = soft_threshold(rhs_ub, lambd)

        if self.obbt:
            obbt_lb, obbt_ub = self.canonicalizer.obbt(rhs_expr)

            step.update_rhs_lb(obbt_lb)
            step.update_rhs_ub(obbt_ub)

            st_obbt_lb = soft_threshold(obbt_lb, lambd)
            st_obbt_ub = soft_threshold(obbt_ub, lambd)

            out_lb = st_obbt_lb
            out_ub = st_obbt_ub
        else:
            out_lb = st_rhs_lb
            out_ub = st_rhs_ub

        self.iterates.append(out_iterate)
        self.lower_bounds[out_iterate] = out_lb
        self.upper_bounds[out_iterate] = out_ub

        # add new iterate and st constraints to canonicalizer
        self.canonicalizer.add_iterate_var(out_iterate, lb=out_lb, ub=out_ub)
        self.canonicalizer.add_soft_threshold_constraints(step, out_lb, out_ub)

        self.add_step(step)

        return out_iterate

    def add_constraint_set(self, constraint_set):
        self.constraint_sets.append(constraint_set)

    def equality_constraint(self, lhs_expr, rhs_expr):
        self.canonicalizer.add_equality_constraint(lhs_expr, rhs_expr)

    # TODO: consider if we should make the theory func local to this func and not a verifier property
    def theory_bound(self, k, target_expr, bound_expr, return_improv_frac=True):
        C = self.theory_func(k)
        bound_lb, bound_ub = self.linear_bound_prop(bound_expr)
        if target_expr.is_leaf:
            self.lower_bounds[target_expr], lb_improv_frac = update_lb(self.lower_bounds[target_expr], bound_lb - C)
            self.upper_bounds[target_expr], ub_improv_frac = update_ub(self.upper_bounds[target_expr], bound_ub + C)
            improv_frac = (lb_improv_frac + ub_improv_frac) / 2.
        self.canonicalizer.add_theory_cut(C, target_expr, bound_lb, bound_ub)

        # TODO: think if we might need to safeguard this with nonleaf theory bounds
        if return_improv_frac:
            return improv_frac

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
        self.canonicalizer.set_zero_objective()

    def set_infinity_norm_objective(self, expr_list):

        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        lb_list = []
        ub_list = []
        for single_expr in expr_list:
            expr_lb, expr_ub = self.linear_bound_prop(single_expr)
            lb_list.append(expr_lb)
            ub_list.append(expr_ub)

        self.canonicalizer.set_infinity_norm_objective(expr_list, lb_list, ub_list)

    def solve(self, **kwargs):
        # print(self.steps)
        res = self.canonicalizer.solve_model(self.steps, self.lower_bounds, self.upper_bounds, **kwargs)
        return res

    def extract_sol(self, iterate):
        return self.canonicalizer.extract_sol(iterate)


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
        raise TypeError(f'lb or ub needs to be an int, float, or np.ndarray. Got {type(val)}')


def relu(v, proj_ranges=None):
    if proj_ranges is None:
        return np.maximum(v, 0)
    else:
        out = v.copy()

        for curr_range in proj_ranges:
            left = curr_range[0]
            right = curr_range[1]
            out[left: right] = np.maximum(v[left: right], 0)

        return out


def saturated_linear(x, l, u):
    return np.minimum(np.maximum(x, l), u)


def soft_threshold(x, gamma):
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def update_lb(old_lb, new_lb):
    n = old_lb.shape[0]
    improv_count = (new_lb > old_lb).sum()
    return np.maximum(old_lb, new_lb), improv_count / n


def update_ub(old_ub, new_ub):
    n = old_ub.shape[0]
    improv_count = (new_ub < old_ub).sum()
    return np.minimum(old_ub, new_ub), improv_count / n


def list_upcast(val):
    # if val is a list then keep it as is otherwise put it in a list
    if isinstance(val, list):
        return val
    else:
        return [val]

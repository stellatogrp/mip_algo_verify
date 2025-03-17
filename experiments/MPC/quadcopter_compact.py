import logging

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation
np.set_printoptions(legacy='1.25')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})

log = logging.getLogger(__name__)


class Quadcopter(object):

    def __init__(self, T=2):
        self.T = T
        Ad = spa.csc_matrix([
        [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
        [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
        [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
        [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
        [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
        [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
        [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
        [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
        [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
        [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
        ])
        Bd = spa.csc_matrix([
        [0.,      -0.0726,  0.,     0.0726],
        [-0.0726,  0.,      0.0726, 0.    ],
        [-0.0152,  0.0152, -0.0152, 0.0152],
        [-0.,     -0.0006, -0.,     0.0006],
        [0.0006,   0.,     -0.0006, 0.0000],
        [0.0106,   0.0106,  0.0106, 0.0106],
        [0,       -1.4512,  0.,     1.4512],
        [-1.4512,  0.,      1.4512, 0.    ],
        [-0.3049,  0.3049, -0.3049, 0.3049],
        [-0.,     -0.0236,  0.,     0.0236],
        [0.0236,   0.,     -0.0236, 0.    ],
        [0.2107,   0.2107,  0.2107, 0.2107]])
        [nx, nu] = Bd.shape
        self.nx = nx
        self.nu = nu

        # Constraints
        u0 = 10.5916
        umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
        umax = np.array([13., 13., 13., 13.]) - u0

        # removing these for size
        # xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
        #                 -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
        # xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
                        # np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        # Objective function
        Q = spa.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
        # QN = Q
        R = 0.1*spa.eye(4)

        self.A, self.B = Ad, Bd
        self.umin, self.umax = umin, umax
        self.Q, self.R = Q, R

        self.xinit = 2 * np.array([1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

        self._create_block_matrices()

    def _create_block_matrices(self):
        nx, nu = self.nx, self.nu
        T = self.T
        H = spa.block_diag([spa.kron(spa.eye(T), self.Q), spa.kron(spa.eye(T-1), self.R)])

        Ax = spa.kron(spa.eye(T), -spa.eye(nx)) + spa.kron(spa.eye(T, k=-1), self.A)
        Ax[:nx, :nx] = spa.eye(nx)
        Bu = spa.kron(spa.vstack([spa.csc_matrix((1, T-1)), spa.eye(T-1)]), self.B)

        Aeq = spa.hstack([Ax, Bu])
        # print(Aeq.shape)
        # Aineq = spa.eye(T * nx + (T-1) * nv)
        Aineq = spa.hstack([spa.csc_matrix(((T-1) * nu, T * nx)), spa.eye((T-1) * nu)])
        M = spa.vstack([Aeq, Aineq])

        self.H = H
        self.M = M

    def test_with_cvxpy(self):
        print('----full cvxpy with all states----')
        H, M = self.H, self.M
        nx, nu, T = self.nx, self.nu, self.T
        n = T * nx + (T - 1) * nu

        xinit = self.xinit
        l = np.zeros(M.shape[0])
        u = np.zeros(M.shape[0])
        l[:nx] = xinit
        u[:nx] = xinit
        l[T * nx:] = np.kron(np.ones(T-1), self.umin)
        u[T * nx:] = np.kron(np.ones(T-1), self.umax)

        x = cp.Variable(n)
        obj = cp.Minimize(.5 * cp.quad_form(x, H))
        constraints = [l <= M @ x, M @ x <= u]
        prob = cp.Problem(obj, constraints)
        res = prob.solve()
        print('res:', res)
        print('xsol:', x.value)


    def test_simplified_cvxpy(self):
        print('----cvxpy with only initial state----')

        nx, nu = self.nx, self.nu
        T = self.T
        A, B = self.A, self.B
        Q, R = self.Q, self.R
        xinit = self.xinit

        P_block = spa.kron(spa.eye(T), Q)
        R_block = spa.kron(spa.eye(T-1), R)

        SX = []
        for i in range(T):
            SX.append(spa.linalg.matrix_power(A, i))
        SX = spa.vstack(SX)

        print('SX shape:', SX.shape)

        SV = spa.csc_matrix(((T-1) * nx, (T-1) * nu))
        for i in range(T-1):
            AiB = spa.linalg.matrix_power(A, i) @ B
            SV += spa.kron(spa.eye(T-1, k=-i), AiB)
        SV = spa.vstack([spa.csc_matrix((nx, (T-1) * nu)), SV])

        print('SV shape:', SV.shape)
        x1 = cp.Variable(nx)
        X = cp.Variable(T * nx)
        V1 = cp.Variable((T-1) * nu)
        obj = cp.Minimize(.5 * cp.quad_form(X, P_block) + .5 * cp.quad_form(V1, R_block))

        umin_stack = np.kron(np.ones(T-1), self.umin)
        umax_stack = np.kron(np.ones(T-1), self.umax)
        constraints = [
            X == SX @ x1 + SV @ V1,
            x1 == xinit,
            umin_stack <= V1, V1 <= umax_stack,
        ]
        prob = cp.Problem(obj, constraints)
        res = prob.solve()
        print(res)
        print('x sol:', X.value)
        print('V sol:', V1.value)

        print('----cvxpy with all things simplified out----')
        X = cp.Variable(nx + (T-1) * nu)
        q = np.zeros(nx + (T-1) * nu)

        H = spa.bmat([
            [SX.T @ P_block @ SX, SX.T @ P_block @ SV],
            [SV.T @ P_block @ SX, SV.T @ P_block @ SV + R_block]
        ])

        l = np.hstack([xinit, umin_stack])
        u = np.hstack([xinit, umax_stack])

        M = spa.eye(nx + (T-1) * nu)

        obj = cp.Minimize(.5 * cp.quad_form(X, H) + q.T @ X)
        constraints = [l <= M @ X, M @ X <= u]
        prob = cp.Problem(obj, constraints)
        res = prob.solve()
        print(res)
        print('x sol:', X.value)

        self.compact_H = H
        self.compact_q = q
        self.compact_M = M
        self.compact_l = l
        self.compact_u = u

        return H, q, M, l, u, X.value


def main():
    qc = Quadcopter(T=10)
    qc.test_with_cvxpy()
    P, q, A, l, u, test_sol = qc.test_simplified_cvxpy()
    rho = 2
    # rho_inv = 1 / rho

    sigma = 1e-4

    m, n = A.shape
    rho = rho * spa.eye(m)

    print('testing with osqp infeas detection formulation')
    xk = np.zeros(n)
    vk = np.zeros(m)

    xv_fp_resids = []
    xv_l2_fp_resids = []
    # xv_rhosigma_resids = []

    K = 50
    # lhs_mat = P + sigma * np.eye(n) + rho * A.T @ A
    lhs_mat = P + sigma * np.eye(n) + A.T @ rho @ A
    for _ in range(K):
        wkplus1 = proj(vk, l, u)
        # xkplus1 = np.linalg.solve(lhs_mat, sigma * xk - q + rho * A.T @ (2 * wkplus1 - vk))
        xkplus1 = np.linalg.solve(lhs_mat, sigma * xk - q + A.T @ rho @ (2 * wkplus1 - vk))
        vkplus1 = vk + A @ xkplus1 - wkplus1

        xv_stack = np.hstack([xkplus1 - xk, vkplus1 - vk])
        xv_fp_resids.append(np.max(np.abs(xv_stack)))
        xv_l2_fp_resids.append(np.linalg.norm(xv_stack))
        # xv_rhosigma_resids.append(np.sqrt(sigma * np.linalg.norm(xkplus1 - xk) ** 2 + rho * np.linalg.norm(wkplus1 - proj(vk, l, u)) ** 2))

        xk = xkplus1
        vk = vkplus1

    print(xk)
    print('norm diff:', np.linalg.norm(xk - test_sol))
    print('xv resids:', xv_fp_resids)
    # print('xv l2 resids:', xv_l2_fp_resids)

    print(np.linalg.norm(test_sol))


def proj(v, l, u):
    return np.minimum(np.maximum(v, l), u)


if __name__ == '__main__':
    main()

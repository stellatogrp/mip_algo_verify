import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation
np.set_printoptions(legacy='1.25')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


class Quadcopter(object):

    def __init__(self, T=2):
        Ad = sparse.csc_matrix([
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
        Bd = sparse.csc_matrix([
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
        Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
        QN = Q
        R = 0.1*sparse.eye(4)

        # Initial and reference states
        x0 = np.ones(12)
        xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

        # Prediction horizon
        N = T

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                            sparse.kron(sparse.eye(N), R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq
        # - input and state constraints

        # simplifying to only have input controls
        # Aineq = sparse.eye((N+1)*nx + N*nu)
        # lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        # uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        # Aineq = sparse.csc_matrix(((N+1)*nx, (N+1)*nx))
        # Aineq = sparse.eye(N*nu)
        Aineq = sparse.hstack([sparse.csc_matrix((N * nu, (N+1)*nx)), sparse.eye(N * nu)])
        lineq = np.kron(np.ones(N), umin)
        uineq = np.kron(np.ones(N), umax)

        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        print(A.shape)
        self.P = P
        self.q = q
        self.A = A
        self.l = l
        self.u = u


def main():
    qc = Quadcopter(T=2)
    P, q, A, l, u = qc.P, qc.q, qc.A, qc.l, qc.u

    x = cp.Variable(A.shape[1])
    obj = .5 * cp.quad_form(x, P) + q.T @ x
    constraints = [l <= A @ x, A @ x <= u]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve(solver=cp.OSQP)
    print(res)
    x = x.value
    print('from cvxpy:', x)

    print('testing with admm osqp formulation')

    m, n = A.shape
    rho = 10
    rho_inv = 1 / rho
    sigma = 1e-4

    xk = np.zeros(n)
    yk = np.zeros(m)
    zk = np.zeros(m)
    sk = zk + rho_inv * yk

    K = 100
    lhs_mat = P + sigma * np.eye(n) + rho * A.T @ A
    xyz_fp_resids = []
    xs_fp_resids = []

    xyz_l2_fp_resids = []
    xs_l2_fp_resids = []
    xz_scaled_resids = []

    for _ in range(K):
        xkplus1 = np.linalg.solve(lhs_mat, sigma * xk - q + rho * A.T @ zk - A.T @ yk)
        zkplus1 = proj(A @ xkplus1 + rho_inv * yk, l, u)
        ykplus1 = yk + rho * (A @ xkplus1 - zkplus1)
        skplus1 = zkplus1 + rho_inv * ykplus1

        xyz_resid = np.hstack([xkplus1 - xk, zkplus1 - zk, ykplus1 - yk])
        xyz_fp_resids.append(np.max(np.abs(xyz_resid)))
        xyz_l2_fp_resids.append(np.linalg.norm(xyz_resid))

        xs_resid = np.hstack([xkplus1 - xk, skplus1 - sk])
        xs_fp_resids.append(np.max(np.abs(xs_resid)))
        xs_l2_fp_resids.append(np.linalg.norm(xs_resid))
        xz_scaled_resids.append(np.sqrt(sigma * np.linalg.norm(xkplus1 - xk) ** 2 + rho * np.linalg.norm(zkplus1 - zk) ** 2))

        xk = xkplus1
        yk = ykplus1
        zk = zkplus1
        sk = skplus1

    print('from admm:', xk)
    print(.5 * xk.T @ P @ xk + q.T @ xk)

    print('xyz resids:', xyz_fp_resids)
    print('xs resids:', xs_fp_resids)
    print('xyz l2 resids:', xyz_l2_fp_resids)
    print('xs l2 resids:', xs_l2_fp_resids)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, K+1), xs_fp_resids, label='infty norm')
    plt.plot(range(1, K+1), xs_l2_fp_resids, label='2 norm')
    plt.plot(range(1, K+1), xz_scaled_resids)
    plt.title('OSQP formulation')

    plt.yscale('log')
    plt.ylabel('residual')
    plt.xlabel(r'$K$')
    plt.legend()
    # plt.show()
    plt.savefig('osqp_version.pdf')

    plt.cla()
    plt.clf()
    plt.close()

    print('testing with osqp infeas detection formulation')
    xk = np.zeros(n)
    vk = np.zeros(m)

    xv_fp_resids = []
    xv_l2_fp_resids = []
    xv_rhosigma_resids = []

    K = 100
    for _ in range(K):
        wkplus1 = proj(vk, l, u)
        xkplus1 = np.linalg.solve(lhs_mat, sigma * xk - q + rho * A.T @ (2 * wkplus1 - vk))
        vkplus1 = vk + A @ xkplus1 - wkplus1

        xv_stack = np.hstack([xkplus1 - xk, zkplus1 - zk])
        xv_fp_resids.append(np.max(np.abs(xv_stack)))
        xv_l2_fp_resids.append(np.linalg.norm(xv_stack))
        xv_rhosigma_resids.append(np.sqrt(sigma * np.linalg.norm(xkplus1 - xk) ** 2 + rho * np.linalg.norm(wkplus1 - proj(zk, l, u)) ** 2))

        xk = xkplus1
        vk = vkplus1

    print(xk)
    print('xv resids:', xv_fp_resids)
    print('xv l2 resids:', xv_l2_fp_resids)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, K+1), xv_fp_resids, label='infty norm')
    plt.plot(range(1, K+1), xv_l2_fp_resids, label='2 norm')
    plt.plot(range(1, K+1), xv_rhosigma_resids)
    plt.title('Fixed point formulation')

    plt.yscale('log')
    plt.ylabel('residual')
    plt.xlabel(r'$K$')
    plt.legend()
    # plt.show()
    plt.savefig('fixed_pt_version.pdf')


def proj(v, l, u):
    return np.minimum(np.maximum(v, l), u)


if __name__ == '__main__':
    main()

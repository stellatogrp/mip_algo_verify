
import cvxpy as cp
import jax
import jax.numpy as jnp

jnp.set_printoptions(precision=5)  # Print few decimal places
jnp.set_printoptions(suppress=True)  # Suppress scientific notation
jax.config.update("jax_enable_x64", True)


class RandomLP(object):

    def __init__(self, m, n, rng_seed=0):
        self.m = m
        self.n = n
        self.key = jax.random.PRNGKey(rng_seed)
        self._generate_data()

    def _generate_data(self):
        self.key, subkey = jax.random.split(self.key)
        self.A = jax.random.normal(subkey, shape=(self.m, self.n))

        self.key, subkey = jax.random.split(self.key)
        self.c = jax.random.uniform(subkey, shape=(self.n,))

        self.key, subkey = jax.random.split(self.key)
        # self.b = jax.random.normal(subkey, shape=(self.m,))
        self.b = jax.random.uniform(subkey, shape=(self.m, ))

    def test_with_cvxpy(self):
        x = cp.Variable(self.n)

        constraints = [self.A @ x == self.b, x >= 0]
        prob = cp.Problem(cp.Minimize(self.c @ x), constraints)
        res = prob.solve()
        print('--testing with cvxpy--')
        print('obj:', res)
        print('x value:', x.value)
        print('y value:', constraints[0].dual_value)

    def test_with_pdhg(self, t=.1, K=10000):
        c, A, b = self.c, self.A, self.b
        m, n = self.m, self.n

        xk = jnp.zeros(n)
        yk = jnp.zeros(m)

        print('--testing with vanilla pdhg--')
        for _ in range(K):
            xkplus1 = jax.nn.relu(xk - t * (c - A.T @ yk))
            ykplus1 = yk - t * (A @ (2 * xkplus1 - xk) - b)

            # print(jnp.linalg.norm(ykplus1 - yk, 1) + jnp.linalg.norm(xkplus1 - xk, 1))

            xk = xkplus1
            yk = ykplus1

        print('obj:', c @ xk)
        print('xvalue:', xk)
        print('yvalue:', yk)


    def test_with_jax_pdhg(self, t=.1, K=10000):
        c, A, b = self.c, self.A, self.b
        m, n = self.m, self.n

        x0 = jnp.zeros(n)
        y0 = jnp.zeros(m)

        print('--testing with jax vanilla pdhg--')
        def body_fun(i, val):
            xk, yk = val
            xkplus1 = jax.nn.relu(xk - t * (c - A.T @ yk))
            ykplus1 = yk - t * (A @ (2 * xkplus1 - xk) - b)
            return (xkplus1, ykplus1)

        xk, yk = jax.lax.fori_loop(0, K, body_fun, (x0, y0))
        print('obj:', c @ xk)
        print('xvalue:', xk)
        print('yvalue:', yk)

    def test_with_pdhg_momentum(self, t=.1, K=10):
        c, A, b = self.c, self.A, self.b
        m, n = self.m, self.n

        xk = jnp.zeros(n)
        yk = jnp.zeros(m)

        # vk = xk
        print('--testing with pdhg plus momentum--')
        # specific momentum from https://arxiv.org/pdf/2403.11139 with Nesterov weights
        for k in range(K):
            xkplus1 = jax.nn.relu(xk - t * (c - A.T @ yk))
            vkplus1 = xkplus1 + k / (k + 3) * (xkplus1 - xk)
            ykplus1 = yk - t * (A @ (2 * vkplus1 - xk) - b)

            xk = xkplus1
            # vk = vkplus1
            yk = ykplus1

        print('obj:', c @ xk)
        print('xvalue:', xk)
        print('yvalue:', yk)


def main():
    m = 5
    n = 10
    K = 100
    instance = RandomLP(m, n)
    instance.test_with_cvxpy()
    instance.test_with_pdhg(K=K)
    instance.test_with_jax_pdhg(K=K)
    instance.test_with_pdhg_momentum(K=K)


if __name__ == '__main__':
    main()

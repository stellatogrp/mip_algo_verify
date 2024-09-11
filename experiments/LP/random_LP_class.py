
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


def main():
    m = 5
    n = 10
    instance = RandomLP(m, n)
    instance.test_with_cvxpy()
    instance.test_with_pdhg(K=20)


if __name__ == '__main__':
    main()

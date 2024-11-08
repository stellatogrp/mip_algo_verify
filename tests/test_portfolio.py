import numpy as np

# import matplotlib.pyplot as plt

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_portfolio():
    n = 3
    d = 2
    # gamma = 2
    # lambd = 1e-4

    np.random.seed(0)
    F = np.random.normal((n, d))
    print(F)

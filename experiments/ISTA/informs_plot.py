import logging

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

jnp.set_printoptions(precision=5)  # Print few decimal places
jnp.set_printoptions(suppress=True)  # Suppress scientific notation
jax.config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def main():
    vp_resid_fname = 'outputs/2024-10-21/15-57-20/resids.csv'
    samples_fname = 'outputs/2024-10-21/15-57-20/sample_resids.csv'
    pep_fname = 'pep_outputs/2024-10-21/16-06-14/pep_resids.csv'

    pep_resids = pd.read_csv(pep_fname, header=None)
    vp_resids = pd.read_csv(vp_resid_fname, header=None)
    sample_resids = pd.read_csv(samples_fname, header=None)
    sample_resids = sample_resids.to_numpy()

    max_sample_resids = np.max(sample_resids, axis=0)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(pep_resids[0])+1), pep_resids[0], label='PEP', color='green', marker='o')
    ax.plot(range(1, len(vp_resids[0])+1), vp_resids[0], label='VP', color='blue', marker='<')
    ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM', color='red', marker='x')

    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Worst case fixed-point residual')
    ax.set_yscale('log')
    ax.set_title(r'ISTA VP with $\infty$-norm')

    plt.legend()
    # plt.show()
    plt.savefig('informs.pdf')


if __name__ == '__main__':
    main()

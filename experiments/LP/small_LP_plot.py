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
#     pep_resid_fname = 'pep_outputs/2024-10-08/12-03-57/pep_resids.csv'
#     pep_time_fname = 'pep_outputs/2024-10-08/12-03-57/pep_times.csv'

#     vp_resid_fname = 'outputs/2024-10-08/10-50-20/vanilla_resids.csv'
#     vp_time_fname = 'outputs/2024-10-08/10-50-20/vanilla_times.csv'

#     sample_resid_fname = 'outputs/2024-10-08/10-50-20/sample_resids.csv'

    pep_resid_fname = 'pep_outputs/2024-10-14/00-42-07/pep_resids.csv'
    pep_time_fname = 'pep_outputs/2024-10-14/00-42-07/pep_times.csv'

    vp_resid_fname = 'outputs/2024-10-13/22-07-27/vanilla_resids.csv'
    vp_time_fname = 'outputs/2024-10-13/22-07-27/vanilla_times.csv'

    sample_resid_fname = 'outputs/2024-10-13/22-07-27/sample_resids.csv'

    m = 11
    n = 14
    pep_resids = pd.read_csv(pep_resid_fname, header=None)
    pep_times = pd.read_csv(pep_time_fname, header=None)
    vp_resids = pd.read_csv(vp_resid_fname, header=None)
    vp_times = pd.read_csv(vp_time_fname, header=None)
    sample_resids = pd.read_csv(sample_resid_fname, header=None)
    sample_resids = sample_resids.to_numpy()

    max_sample_resids = np.max(sample_resids, axis=0)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(pep_resids[0])+1), pep_resids[0] / np.sqrt(n + m), label='Scaled PEP')
    ax.plot(range(1, len(max_sample_resids)+1), max_sample_resids, label='SM', linewidth=5, alpha=0.3)
    ax.plot(range(1, len(vp_resids[0])+1), vp_resids[0], label='VP')

    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Fixed-point residual')
    ax.set_yscale('log')
    ax.set_title(rf'PDHG VP, $m={m}$, $n={n}$')

    ax.legend()

    plt.tight_layout()

    plt.savefig('small_pdhg_resids.pdf')

    plt.clf()
    plt.cla()
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(range(1, len(pep_times[0])+1), pep_times[0], label='PEP')
    ax.plot(range(1, len(vp_times[0])+1), vp_times[0], label='VP')

    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Solvetime (s)')
    ax.set_yscale('log')
    ax.set_title(rf'PDHG VP, $m={m}$, $n={n}$')

    ax.legend()

    plt.tight_layout()

    plt.savefig('small_pdhg_times.pdf')

    plt.clf()
    plt.cla()
    plt.close()

if __name__ == '__main__':
    main()

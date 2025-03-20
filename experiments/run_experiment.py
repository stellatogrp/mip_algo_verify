import logging
import os
import sys

import FISTA.FISTA as FISTA
import hydra
import ISTA.ISTA as ISTA

# import LP.LP as LP
import LP.LP_satlin as LP

# import MPC.MPC as MPC
import MPC.MPC_noxinit as MPC

log = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path='configs/ISTA', config_name='ista_experiment.yaml')
def main_experiment_fista(cfg):
    FISTA.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/ISTA', config_name='ista_experiment.yaml')
def main_experiment_ista(cfg):
    ISTA.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/LP', config_name='lp_experiment.yaml')
def main_experiment_lp(cfg):
    # if cfg.incremental:
    #     LP_incr.run(cfg)
    # else:
    #     LP.run(cfg)
    LP.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/MPC', config_name='mpc_experiment.yaml')
def main_experiment_mpc(cfg):
    MPC.run(cfg)


base_dir_map = {
    'LP': 'LP/outputs',
    'MPC': 'MPC/outputs',
    'NNQP': 'NNQP/outputs',
    'ISTA': 'ISTA/outputs',
    'ISTA_scratch': 'ISTA_scratch/outputs',
    'FISTA': 'FISTA/outputs',
    'Portfolio': 'Portfolio/outputs',
}


func_driver_map = {
    'LP': main_experiment_lp,
    'MPC': main_experiment_mpc,
    # 'NNQP': main_experiment_nnqp,
    'ISTA': main_experiment_ista,
    # 'ISTA_scratch': main_experiment_ista_scratch,
    'FISTA': main_experiment_fista,
    # 'Portfolio': main_experiment_portfolio,
}


LP_params = [
    ['momentum=False', 'K_max=50', 'huchette_cuts=True'],
    ['momentum=False', 'K_max=50', 'huchette_cuts=False'],
    ['momentum=True', 'K_max=50', 'huchette_cuts=True'],
    ['momentum=True', 'K_max=50', 'huchette_cuts=False'],
]

ISTA_params = [
    ['m=25', 'n=20', 'huchette_cuts=True'],
    ['m=20', 'n=25', 'huchette_cuts=True'],
    ['m=25', 'n=20', 'huchette_cuts=False'],
    ['m=20', 'n=25', 'huchette_cuts=False'],
]

# add FISTA params
FISTA_params = [
    ['m=25', 'n=20', 'huchette_cuts=True'],
    ['m=20', 'n=25', 'huchette_cuts=True'],
    ['m=25', 'n=20', 'huchette_cuts=False'],
    ['m=20', 'n=25', 'huchette_cuts=False'],
]

MPC_params = [
    ['rho=1'],
    ['rho=10'],
    ['rho=100'],
]

def main():
    if len(sys.argv) < 3:
        print('not enough command line arguments')
        exit(0)
    if sys.argv[2] == 'cluster':
        # raise NotImplementedError
        base_dir = '/scratch/gpfs/vranjan/mip_algo_verify_out'
    elif sys.argv[2] == 'local':
        base_dir = '.'
    else:
        print('specify cluster or local')
        exit(0)

    experiment = sys.argv[1]
    target_machine = sys.argv[2]

    if sys.argv[1] not in base_dir_map:
        print(f'experiment name "{sys.argv[1]}" invalid')
        exit(0)

    base_dir = f'{base_dir}/{base_dir_map[sys.argv[1]]}'
    driver = func_driver_map[sys.argv[1]]

    if target_machine == 'local' or "SLURM_ARRAY_TASK_ID" not in os.environ:
        hydra_tags = [f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}', 'hydra.job.chdir=True']
    else:
        job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        log.info(f'job id: {job_idx}')
        hydra_tags = [f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}_{job_idx}', 'hydra.job.chdir=True']

        if experiment == 'LP':
            hydra_tags += LP_params[job_idx]

        if experiment == 'MPC':
            hydra_tags += MPC_params[job_idx]

        if experiment == 'ISTA':
            hydra_tags += ISTA_params[job_idx]

        if experiment == 'FISTA':
            hydra_tags += FISTA_params[job_idx]

    sys.argv = [sys.argv[0]] + hydra_tags

    driver()


if __name__ == '__main__':
    main()

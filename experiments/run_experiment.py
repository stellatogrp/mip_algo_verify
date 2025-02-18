import logging
import os
import sys

import hydra
import ISTA.ISTA as ISTA
import LP.LP as LP

log = logging.getLogger(__name__)


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


base_dir_map = {
    'LP': 'LP/outputs',
    'NNQP': 'NNQP/outputs',
    'ISTA': 'ISTA/outputs',
    'ISTA_scratch': 'ISTA_scratch/outputs',
    'FISTA': 'FISTA/outputs',
    'Portfolio': 'Portfolio/outputs',
}


func_driver_map = {
    'LP': main_experiment_lp,
    # 'NNQP': main_experiment_nnqp,
    'ISTA': main_experiment_ista,
    # 'ISTA_scratch': main_experiment_ista_scratch,
    # 'FISTA': main_experiment_fista,
    # 'Portfolio': main_experiment_portfolio,
}


# NNQP_params = [
#     ['n=10', 'two_step=True', 'one_step=False'],
#     ['n=10', 'two_step=False', 'one_step=True'],
#     ['n=20', 'two_step=True', 'one_step=False'],
#     ['n=20', 'two_step=False', 'one_step=True'],
#     ['n=30', 'two_step=True', 'one_step=False'],
#     ['n=30', 'two_step=False', 'one_step=True'],
#     ['n=40', 'two_step=True', 'one_step=False'],
#     ['n=40', 'two_step=False', 'one_step=True'],
# ]

# LP_params = [
#     ['flow.x.demand_ub=-6', 'momentum=False'],
#     ['flow.x.demand_ub=-6', 'momentum=True'],
#     ['flow.x.demand_ub=-5', 'momentum=False'],
#     ['flow.x.demand_ub=-5', 'momentum=True'],
#     ['flow.x.demand_ub=-4', 'momentum=False'],
#     ['flow.x.demand_ub=-4', 'momentum=True'],
#     ['flow.x.demand_ub=-3', 'momentum=False'],
#     ['flow.x.demand_ub=-3', 'momentum=True'],
#     ['flow.x.demand_ub=-2', 'momentum=False'],
#     ['flow.x.demand_ub=-2', 'momentum=True'],
# ]

# LP_params = [
#     ['momentum=False', 'K_max=50', 'mipfocus=0'],
#     # ['momentum=False', 'K_max=50', 'mipfocus=3'],
#     ['momentum=True', 'K_max=50', 'mipfocus=0'],
#     # ['momentum=True', 'K_max=50', 'mipfocus=3'],
# ]

LP_params = [
    ['momentum=False', 'K_max=50', 'huchette_cuts=True'],
    ['momentum=False', 'K_max=50', 'huchette_cuts=False'],
    ['momentum=True', 'K_max=50', 'huchette_cuts=True'],
    ['momentum=True', 'K_max=50', 'huchette_cuts=False'],
]

ISTA_params = [
    ['m=20', 'n=15', 'K_max=50', 'lambd.val=0.01'],
    ['m=15', 'n=20', 'K_max=50', 'lambd.val=0.01'],
    # ['m=30', 'n=20', 'K_max=50', 'lambd.val=1'],
    # ['m=20', 'n=30', 'K_max=50', 'lambd.val=1'],
]

ISTA_scratch_params = [
    ['m=30', 'n=20', 'K_max=25'],
    ['m=20', 'n=30', 'K_max=25'],
]

# add FISTA params
FISTA_params = [
    ['m=20', 'n=15', 'K_max=40', 'lambd.val=0.01'],
    ['m=15', 'n=20', 'K_max=40', 'lambd.val=0.01'],
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

        if experiment == 'ISTA':
            hydra_tags += ISTA_params[job_idx]

        if experiment == 'FISTA':
            hydra_tags += FISTA_params[job_idx]

    sys.argv = [sys.argv[0]] + hydra_tags

    driver()


if __name__ == '__main__':
    main()

import logging
import os
import sys

import hydra
import ISTA.ISTA_incremental as ISTA_incr
import LP.LP as LP
import LP.LP_incremental as LP_incr
import NNQP.NNQP as NNQP
import NNQP.NNQP_vec as NNQP_vec

log = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path='configs/NNQP', config_name='nnqp_experiment.yaml')
def main_experiment_nnqp(cfg):
    if cfg.vec:
        log.info('NNQP vec')
        NNQP_vec.run(cfg)
    else:
        log.info('NNQP')
        NNQP.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/LP', config_name='lp_experiment.yaml')
def main_experiment_lp(cfg):
    if cfg.incremental:
        LP_incr.run(cfg)
    else:
        LP.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/ISTA', config_name='ista_experiment.yaml')
def main_experiment_ista(cfg):
    ISTA_incr.run(cfg)


base_dir_map = {
    'LP': 'LP/outputs',
    'NNQP': 'NNQP/outputs',
    'ISTA': 'ISTA/outputs',
}


func_driver_map = {
    'LP': main_experiment_lp,
    'NNQP': main_experiment_nnqp,
    'ISTA': main_experiment_ista,
}


NNQP_params = [
    ['n=10', 'two_step=True', 'one_step=False'],
    ['n=10', 'two_step=False', 'one_step=True'],
    ['n=20', 'two_step=True', 'one_step=False'],
    ['n=20', 'two_step=False', 'one_step=True'],
    ['n=30', 'two_step=True', 'one_step=False'],
    ['n=30', 'two_step=False', 'one_step=True'],
    ['n=40', 'two_step=True', 'one_step=False'],
    ['n=40', 'two_step=False', 'one_step=True'],
]

# LP_params = [
#     ['x.l=0', 'momentum=True'],
#     ['x.l=0', 'momentum=False'],
#     ['x.l=0.25', 'momentum=True'],
#     ['x.l=0.25', 'momentum=False'],
#     ['x.l=0.5', 'momentum=True'],
#     ['x.l=0.5', 'momentum=False'],
#     ['x.l=0.75', 'momentum=True'],
#     ['x.l=0.75', 'momentum=False'],
# ]

LP_params = [
    ['x.l=0.75', 'incremental=True'],
    ['x.l=0.75', 'incremental=False'],
    ['x.l=0.75', 'incremental=True', 'm=10', 'n=20', 'K_max=75'],
    ['x.l=0.75', 'incremental=False', 'm=10', 'n=20', 'K_max=75'],
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

        if experiment == 'NNQP':
            hydra_tags += NNQP_params[job_idx]

        if experiment == 'LP':
            hydra_tags += LP_params[job_idx]

    sys.argv = [sys.argv[0]] + hydra_tags

    # if experiment == 'NNQP':  # can use this with the cluster environments to override vars
    #     sys.argv.append('n=15')

    driver()

    # if sys.argv[1] == 'BoxQP':
    #     base_dir = os.path.join(base_dir, 'BoxQP/outputs')
    #     hydra_tags = [f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}', 'hydra.job.chdir=True']
    #     sys.argv = [sys.argv[0]] + hydra_tags
    #     main_experiment_boxqp()


if __name__ == '__main__':
    main()

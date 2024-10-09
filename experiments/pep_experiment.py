import logging
import os
import sys

import hydra
import LP.LP_pep as LP_pep
import NNQP.NNQP_pep as NNQP_pep

log = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path='configs/NNQP', config_name='nnqp_experiment.yaml')
def main_experiment_nnqp(cfg):
    NNQP_pep.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/LP', config_name='lp_experiment.yaml')
def main_experiment_lp(cfg):
    LP_pep.run(cfg)


base_dir_map = {
    'NNQP': 'NNQP/pep_outputs',
    'LP': 'LP/pep_outputs',
}


func_driver_map = {
    'NNQP': main_experiment_nnqp,
    'LP': main_experiment_lp,
}


NNQP_params = [
    ['n=20', 'two_step=True', 'one_step=False'],
    ['n=20', 'two_step=False', 'one_step=True'],
    ['n=30', 'two_step=True', 'one_step=False'],
    ['n=30', 'two_step=False', 'one_step=True'],
    ['n=40', 'two_step=True', 'one_step=False'],
    ['n=40', 'two_step=False', 'one_step=True'],
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

        # if experiment == 'LP':
        #     hydra_tags += LP_params[job_idx]

    sys.argv = [sys.argv[0]] + hydra_tags

    driver()


if __name__ == '__main__':
    main()

import logging
import sys

import hydra
import LP.LP as LP
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
    LP.run(cfg)


base_dir_map = {
    'LP': 'LP/outputs',
    'NNQP': 'NNQP/outputs',
}


func_driver_map = {
    'LP': main_experiment_lp,
    'NNQP': main_experiment_nnqp,
}


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

    if sys.argv[1] not in base_dir_map:
        print(f'experiment name "{sys.argv[1]}" invalid')
        exit(0)

    base_dir = f'{base_dir}/{base_dir_map[sys.argv[1]]}'
    driver = func_driver_map[sys.argv[1]]
    hydra_tags = [f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}', 'hydra.job.chdir=True']
    sys.argv = [sys.argv[0]] + hydra_tags
    driver()

    # if sys.argv[1] == 'BoxQP':
    #     base_dir = os.path.join(base_dir, 'BoxQP/outputs')
    #     hydra_tags = [f'hydra.run.dir={base_dir}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}', 'hydra.job.chdir=True']
    #     sys.argv = [sys.argv[0]] + hydra_tags
    #     main_experiment_boxqp()


if __name__ == '__main__':
    main()

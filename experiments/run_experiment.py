import logging
import os
import sys

import hydra
import ISTA.FISTA_hydra as FISTA_hydra
import ISTA.ISTA_hydra as ISTA_hydra

# import ISTA.ISTA_incremental as ISTA_incr
import ISTA.ISTA_scratch_incremental as ISTA_scratch
import LP.LP as LP

# import LP.LP_incremental as LP_incr
import LP.LP_rel_incremental as LP_incr
import NNQP.NNQP as NNQP
import NNQP.NNQP_vec as NNQP_vec
import Portfolio.PortfolioL1 as PortfolioL1
import Portfolio.PortfolioL2 as PortfolioL2

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
    ISTA_hydra.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/ISTA', config_name='ista_scratch_experiment.yaml')
def main_experiment_ista_scratch(cfg):
    ISTA_scratch.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/ISTA', config_name='ista_experiment.yaml')
def main_experiment_fista(cfg):
    FISTA_hydra.run(cfg)


@hydra.main(version_base='1.2', config_path='configs/Portfolio', config_name='portfolio_experiment.yaml')
def main_experiment_portfolio(cfg):
    if cfg.reg_type == 'l2':
        PortfolioL2.run(cfg)
    elif cfg.reg_type == 'l1':
        PortfolioL1.run(cfg)


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
    'NNQP': main_experiment_nnqp,
    'ISTA': main_experiment_ista,
    'ISTA_scratch': main_experiment_ista_scratch,
    'FISTA': main_experiment_fista,
    'Portfolio': main_experiment_portfolio,
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

LP_params = [
    ['momentum=False', 'K_max=50', 'mipfocus=0'],
    ['momentum=False', 'K_max=50', 'mipfocus=3'],
    ['momentum=True', 'K_max=50', 'mipfocus=0'],
    ['momentum=True', 'K_max=50', 'mipfocus=3'],
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

Portfolio_params = [
    [],
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

        if experiment == 'ISTA':
            hydra_tags += ISTA_params[job_idx]

        if experiment == 'ISTA_scratch':
            hydra_tags += ISTA_scratch_params[job_idx]

        if experiment == 'FISTA':
            hydra_tags += FISTA_params[job_idx]

        if experiment == 'PortfolioL2':
            hydra_tags += Portfolio_params[job_idx]

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

#!/bin/bash
#SBATCH --job-name=LP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=3G
#SBATCH --time=02-12:59:59
#SBATCH --array=0-3
#SBATCH -o /scratch/gpfs/vranjan/mip_algo_verify_out/LP/runs/%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vranjan@princeton.edu
# #SBATCH --gres=gpu:1

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='true'
# export XLA_PYTHON_CLIENT_MEM_FRACTION='0.30'
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export xla_force_host_platform_device_count=1

module purge
module load gurobi/11.0.3
module load anaconda3/2024.6
# module load anaconda3/2023.9 cudnn/cuda-11.x/8.2.0 cudatoolkit/11.3 nvhpc/21.5
conda activate algover

python run_experiment.py LP cluster

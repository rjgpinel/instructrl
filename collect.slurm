#!/bin/bash
#SBATCH --job-name=instructrl-data
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
# #SBATCH --qos=qos_gpu-dev
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
##SBATCH --time=02:00:00
# #SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/pvn/uqn73qm/logs/%j.out
#SBATCH --error=/gpfswork/rech/pvn/uqn73qm/logs/%j.out
# #SBATCH --array=0-49
# # SBATCH --array=0-105
# #SBATCH --array=0-423
# #SBATCH --array=0-529

set -x
set -e

module purge
module load singularity
export PYTHONPATH=/opt/YARR/
export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID
mkdir $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
pwd; hostname; date

log_dir=$WORK/logs

mkdir -p $log_dir

rm /tmp/.X99-lock || true

export PROJECT_DIR="$WORK/Code/instructrl"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/instructrl/models"
echo $PYTHONPATH

SAVE_PATH=$DATASET/instructrl_bis/

mkdir -p $SAVE_PATH

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	--cpus-per-task 1 \
	singularity exec --nv \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE,/gpfslocalsup:/gpfslocalsup/,/gpfslocalsys:/gpfslocalsys,/gpfs7kw:/gpfs7kw,/gpfsssd:/gpfsssd,/gpfsdsmnt:/gpfsdsmnt,/gpfsdsstore:/gpfsdsstore \
	$SINGULARITY_ALLOWED_DIR/vlc_rlbench.sif \
	xvfb-run -a \
		-e $log_dir/${SLURM_JOBID}.out \
	  /usr/bin/python3.9 $PROJECT_DIR/data/collect_data.py \
    --save_path $SAVE_PATH

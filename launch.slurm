#!/bin/bash
#SBATCH --job-name=vlm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
# #SBATCH --qos=qos_gpu-dev
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
# #SBATCH --time=10:00:00
# #SBATCH --time=02:00:00
#SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/pvn/uqn73qm/logs/%j.out
#SBATCH --error=/gpfswork/rech/pvn/uqn73qm/logs/%j.out
#SBATCH --array=0-49
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

alhistory=$WORK/Code/hiveformer-theo/
xp=${xp:-$WORK/xp_theo/}
log_dir=$WORK/logs
seed=${seed:-0}
tasks=${tasks:-$WORK/Code/hiveformer/10-tasks.csv}
num_seeds=${num_seeds:-5}
# tasks=${tasks:-push_button}
# tasks=${tasks:-$WORK/Code/hiveformer/tasks.csv}

if [ ! -z $SLURM_ARRAY_TASK_ID ]; then
  if [ -f $tasks ]; then
  	task_file=$tasks
  	num_tasks=$(wc -l < $task_file)
  	task_id=$(( (SLURM_ARRAY_TASK_ID % num_tasks) +1 ))
  	tasks=$(sed -n "${task_id},${task_id}p" $tasks)
  else
  	num_tasks="${#tasks[@]}"
  fi
  seed=$(( SLURM_ARRAY_TASK_ID  / num_tasks ))
fi

dataset=${dataset:-$DATASET/datasets/hiveformer/dataset-$seed}
name=${name:-${prefix}${tasks}}

if [ ! -z $num_seeds ]; then
	valset=${valset:-$DATASET/datasets/hiveformer/dataset-$((($seed + 1) % $num_seeds))}
fi


mkdir -p $log_dir
mkdir -p $xp

rm /tmp/.X99-lock || true

instructions=$WORK/Code/hiveformer/instructions-clip.pkl

srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
	--cpus-per-task 1 \
	singularity exec --nv \
	--bind $WORK:$WORK,$SCRATCH:$SCRATCH,$STORE:$STORE,/gpfslocalsup:/gpfslocalsup/,/gpfslocalsys:/gpfslocalsys,/gpfs7kw:/gpfs7kw,/gpfsssd:/gpfsssd,/gpfsdsmnt:/gpfsdsmnt,/gpfsdsstore:/gpfsdsstore \
	$SINGULARITY_ALLOWED_DIR/hiveformer.sif \
	xvfb-run -a \
		-e $log_dir/${SLURM_JOBID}.out \
	  /usr/bin/python3.9 $alhistory/train_without_rlbench.py \
		--name $name \
		--seed $seed \
		--tasks $tasks \
		--dataset $dataset \
    --instructions $instructions \
		--xp $xp \
    --valset $valset \
		--headless \
		$args

export PROJECT_DIR="$WORK/Code/instructrl"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/instructrl/models"
echo $PYTHONPATH
export WANDB_API_KEY='d1e8f69de29481f3793656cb29f35e9c2b53e812'

export bucket_name='instruct-rl'
export experiment_name='instructrl'

ONLINE=True
DATASET="reach_target"
MODEL_TYPE="vit_base"
TRANSFER_TYPE="m3ae_vit_b16"
BATCH_SIZE=2048
INSTRUCTION="moving to one of the spheres"
NOTE="pt: $TRANSFER_TYPE inst: $INSTRUCTION batch size: $BATCH_SIZE policy: $MODEL_TYPE dataset: $DATASET"

python3 -m instructrl.instructrl_main \
    --is_tpu=True \
    --dataset_name="$DATASET" \
    --model.model_type="$MODEL_TYPE" \
    --model.transfer_type="$TRANSFER_TYPE" \
    --window_size=4 \
    --val_every_epochs=1 \
    --test_every_epochs=1 \
    --instruct="$INSTRUCTION" \
    --batch_size="$BATCH_SIZE" \
    --weight_decay=0.0 \
    --lr=3e-4 \
    --auto_scale_lr=False \
    --lr_schedule=cos \
    --warmup_epochs=5 \
    --momentum=0.9 \
    --clip_gradient=10.0 \
    --epochs=200 \
    --dataloader_n_workers=16 \
    --dataloader_shuffle=False \
    --log_all_worker=False \
    --logging.online="$ONLINE" \
    --logging.prefix='' \
    --logging.project="$experiment_name" \
    --logging.gcs_output_dir="gs://$bucket_name/instructrl/experiment_output/$experiment_name" \
    --logging.output_dir="$HOME/experiment_output/$experiment_name" \
    --logging.random_delay=0.0 \
    --logging.notes="$NOTE"

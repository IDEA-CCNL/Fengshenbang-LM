#!/bin/bash
#SBATCH --job-name=t5vae_pretran
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH -o outputs/t5_vae/job_out/%x-%j.log
#SBATCH -e outputs/t5_vae/job_out/%x-%j.err
## SBATCH --requeue
## SBATCH --qos=preemptive

set -x -e

ulimit -s unlimited
echo "START TIME: $(date)"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_ADDR=127.0.0.1
MASTER_PORT=$[RANDOM%10000+50000]

GPUS_PER_NODE=8
NNODES=2   # switch to 128
TP_SIZE=1    # always fixed to the size of a single node
PP_SIZE=1    # NLAYERS must be a multiple of PP_SIZE here

MICRO_BATCH_SIZE=18
ZERO_STAGE=2

ROOT_PATH=/cognitive_comp/gaoxinyu/ln_model/t5_vae
config_json=./ds_config.json

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
export PL_DEEPSPEED_CONFIG_PATH=$config_json

VAE_YAML=./vae_config.yaml

MODEL_PATH=/cognitive_comp/wanghao/models/mengzi-t5-base
T5_VAE_ARGS="
    --vae_model_path $MODEL_PATH \
    --vae_config $VAE_YAML \
"
CHECKPOINT_SAVE_PATH=${ROOT_PATH}/checkpoints
MODEL_CHECKPOINT_ARGS="\
        --monitor val_loss \
        --save_top_k -1 \
        --mode min \
        --every_n_train_steps 10000 \
        --save_weights_only True \
        --dirpath $CHECKPOINT_SAVE_PATH \
        --filename checkpoint-{epoch:2d}-{step:7d} \
        "

TRAINER_ARGS="
    --max_epochs 100 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy deepspeed_stage_2 \
    --precision 16 \
    --val_check_interval 1000 \
    --default_root_dir ${ROOT_PATH} \
"

corpora_path=/cognitive_comp/wanghao/datasets/t5_vae/wudao
train_datas=`find ${corpora_path}/* -type d -name "00*" | awk -F "\/" '{path=$7"/train";print path}' | paste -sd " "`
val_datas=`find ${corpora_path}/*  -type d -name "00*" | awk -F "\/" '{path=$7"/val";print path}' | paste -sd " "`
test_datas=`find ${corpora_path}/*  -type d -name "00*" | awk -F "\/" '{path=$7"/test";print path}' | paste -sd " "`

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --eval_batchsize $MICRO_BATCH_SIZE \
    --test_batchsize $MICRO_BATCH_SIZE \
    --datas_common_prefix ${corpora_path}
    --train_datas $train_datas \
    --valid_datas $val_datas \
    --test_datas $test_datas \
    --input_tensor_name input_ids attn_mask decoder_target \
"

# export LAUNCHER="python -u -m torch.distributed.launch \
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --max_restarts 0 \
#     "

SCRIPTS_PATH=/cognitive_comp/gaoxinyu/FS/fengshen_wh/fengshen

export CMD=" \
    $SCRIPTS_PATH/pretrain_t5_vae.py \
    $TRAINER_ARGS \
    $MODEL_CHECKPOINT_ARGS \
    $T5_VAE_ARGS \
    $DATA_ARGS \
    "

# echo $CMD

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
#singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH $LAUNCHER --node_rank 0 $CMD

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
# clear; srun --jobid $SLURM_JOBID singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
#clear; srun --jobid $SLURM_JOBID singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python $CMD'
CUDA_VISIBLE_DEVICES=7 singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python -m debugpy --listen 192.168.190.2:51000 --wait-for-client $CMD'
# clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
# clear; srun --jobid $SLURM_JOBID bash -c 'python $CMD'
# clear; bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

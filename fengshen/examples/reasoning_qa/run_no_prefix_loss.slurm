#!/bin/bash
  
#SBATCH -J reasoning_math
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:6
#SBATCH -o logs/reasoning_math-gptj-%j.log
#SBATCH -e errs/reasoning_math-gptj-%j.err

# set -x -e
export MASTER_PORT=$[RANDOM%10000+50000]
export TORCH_EXTENSIONS_DIR=/home/zhuxinyu/.cache/torch_extensions/py38_cu111
NUM_NODES=1
NUM_GPUS=8

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M")"
ROOT_DIR=/cognitive_comp/zhuxinyu/codes/reasoning_qa

ZERO_STAGE=2

config_json="$ROOT_DIR/ds_config.$SLURM_JOBID.json"
# config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_micro_batch_size()
        # "offload_param":{
        #   "device":"cpu",
        #   "pin_memory":true
        # },
cat <<EOT > $config_json
{
    "train_micro_batch_size_per_gpu":1,
    "steps_per_print":100,
    "gradient_clipping":1,
    "zero_optimization":{
        "stage": $ZERO_STAGE,
        "offload_optimizer":{
          "device":"cpu",
          "pin_memory":true
        },
        "overlap_comm":true,
        "contiguous_gradients":true,
        "sub_group_size":1000000000,
        "stage3_max_live_parameters":1000000000,
        "stage3_max_reuse_distance":1000000000,
        "stage3_gather_fp16_weights_on_model_save":true
    },
    "optimizer":{
        "type":"Adam",
        "params":{
            "lr": 5e-7,
            "weight_decay":0
        }
    },
    "scheduler":{
        "type":"WarmupLR",
        "params":{
            "warmup_min_lr":1e-7,
            "warmup_max_lr":5e-7
        }
    },
    "zero_allow_untested_optimizer":false,
    "fp16":{
        "enabled":true,
        "loss_scale":0,
        "loss_scale_window":1000,
        "hysteresis":2,
        "min_loss_scale":1
    },
    "activation_checkpointing":{
        "partition_activations":false,
        "contiguous_memory_optimization":false
    },
    "wall_clock_breakdown":false
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$config_json

TRAINER_ARGS="
    --max_epochs 20 \
    --gpus $NUM_GPUS \
    --log_every_n_steps 1 \
    --precision 16 \
    --save_dir $ROOT_DIR/outputs \
    --save_top_k 5 \
    --monitor avg_val_loss \
    --mode min \
    --timestamp $TIMESTAMP \
    --gradient_clip_val 1.0 \
    --strategy deepspeed_stage_2 \
    --train \
    --test \
    --num_nodes $NUM_NODES \
"
    # --strategy ddp \
    # --val_check_interval 0.5 \
    # --patience 3 \
    # --check_val_every_n_epoch 1 \

DATA_DIR=/cognitive_comp/zhuxinyu/datasets/grade-school-math/grade_school_math/data/
DATA_ARGS="
    --data_dir $DATA_DIR \
    --train_data train.jsonl \
    --valid_data test.jsonl \
    --test_data test.jsonl \
    --num_workers 32 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --valid_batch_size 1 \
    --test_batch_size 1 \
"
    # --recreate_dataset \
    # --source_max_token_len 512 \

MODEL_ARGS="
    --seed 19990303 \
    --model_type gpt \
    --model_name /cognitive_comp/zhuxinyu/pretrained_models/EleutherAI/gpt-j-6B \
    --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-04-20_07-44/last-epoch=11-avg_val_loss=0.8180.ckpt \
    --lr 5e-7 \
    --l2 0. \
    --warmup 0.1 \
    --show_training_ex 100 \
    --scheduler linear \
"
    # --loss_on_prefix \

    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-04-17_18-26-mapper_openai/last-epoch=08-avg_val_loss=1.0418.ckpt \
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-04-18_15-45/last-epoch=08-avg_val_loss=1.1399.ckpt \
    # --model_name gpt-j-6B \
    # --text \


SCRIPTS_PATH=$ROOT_DIR/gpt_training.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
# --jobid $SLURM_JOBID
# --ntasks-per-node=8
# --cpus-per-gpu=32
srun  --nodes=$NUM_NODES --gres=gpu:$NUM_GPUS --cpus-per-gpu=32 bash -c 'python $CMD'
# bash -c 'python $CMD'

# SINGULARITY_PATH=/cognitive_comp/zhuxinyu/container/pytorch21_06_py3_docker_image_v2.sif
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python $CMD'

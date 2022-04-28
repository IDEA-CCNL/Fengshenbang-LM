#!/bin/bash
#SBATCH --job-name=t5_cn_small_pretrain_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err
#SBATCH -x dgx050

set -x -e
source activate base

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=32
ROOT_DIR=/cognitive_comp/ganruyi/experiments/t5_cn_small_pretrain_v2/

ZERO_STAGE=1

config_json="$ROOT_DIR/ds_config.t5_cn_small_pretrain_v2.$SLURM_JOBID.json"
export MASTER_PORT=$[RANDOM%10000+30000]
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()

cat <<EOT > $config_json
{
    "zero_optimization": {
        "stage": 1
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "params": {
            "betas": [
                0.9,
                0.95
            ],
            "eps": 1e-08,
            "lr": 1e-04,
            "weight_decay": 0.01
        },
        "type": "AdamW"
    },
    "scheduler": {
        "type": "WarmupLR",
        "params":{
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 10000
        }
    },
    "steps_per_print": 100,
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "zero_allow_untested_optimizer": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions
# strategy=ddp
strategy=deepspeed_stage_1

TRAINER_ARGS="
    --max_epochs 1 \
    --gpus 8 \
    --num_nodes 1 \
    --strategy ${strategy} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --every_n_train_steps 0 \
    --monitor train_loss \
    --mode min \
    --save_last \
    --val_check_interval 0.01 \
    --preprocessing_num_workers 20 \
"
# --accumulate_grad_batches 8 \
DATA_DIR=wudao_180g_mt5_tokenized

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --train_data ${DATA_DIR} \
    --train_split_size 0.999 \
    --max_seq_length 1024 \
"

MODEL_ARGS="
    --pretrained_model_path /cognitive_comp/ganruyi/experiments/t5_cn_small_pretrain/Randeng-T5-77M \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --keep_tokens_path /cognitive_comp/ganruyi/hf_models/t5_cn_small/sentencepiece_cn_keep_tokens.json \
"
# --resume_from_checkpoint /cognitive_comp/ganruyi/fengshen/t5_cn_small_pretrain/ckpt/last.ckpt \

SCRIPTS_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/examples/pretrain_t5/pretrain_t5.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD

SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
# salloc --nodes=1 --gres=gpu:2 --cpus-per-gpu=20 -t 24:00:00
clear; srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c '/home/ganruyi/anaconda3/bin/python $CMD'
# clear; srun --job-name=t5_cn_small_pretrain_v2 --jobid=153124 --nodes=1 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=30 -o %x-%j.log -e %x-%j.err singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c '/home/ganruyi/anaconda3/bin/python $CMD'

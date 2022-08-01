#!/bin/bash
  
#SBATCH -J csqa
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:4
#SBATCH -o logs/csqa-gpt2-large-%j.log
#SBATCH -e errs/csqa-gpt2-large-%j.err

# set -x -e
export MASTER_PORT=$[RANDOM%10000+50000]
export TORCH_EXTENSIONS_DIR=/home/zhuxinyu/.cache/torch_extensions/py38_cu111
NUM_NODES=1
NUM_GPUS=2

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M")"
ROOT_DIR=/cognitive_comp/zhuxinyu/codes/reasoning_qa

# ZERO_STAGE=1

# config_json="$ROOT_DIR/ds_config.$SLURM_JOBID.json"
# config_json="./ds_config.json"

# cat <<EOT > $config_json
# {
#     "zero_optimization": {
#         "stage": ${ZERO_STAGE}
#     },
#     "fp16": {
#         "enabled": true,
#         "loss_scale": 0,
#         "loss_scale_window": 1000,
#         "initial_scale_power": 16,
#         "hysteresis": 2,
#         "min_loss_scale": 1
#     },
#     "optimizer": {
#         "params": {
#             "betas": [
#                 0.9,
#                 0.999
#             ],
#             "eps": 1e-08,
#             "lr": 1e-05,
#             "weight_decay": 1e-1
#         },
#         "type": "AdamW"
#     },
#     "scheduler": {
#         "params": {
#             "warmup_max_lr": 1e-05,
#             "warmup_min_lr": 1e-06,
#             "total_num_steps": 800000,
#             "warmup_num_steps" : 8000
#         },
#         "type": "WarmupDecayLR"
#     },
#     "steps_per_print": 100,
#     "gradient_clipping": 1,
#     "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
#     "zero_allow_untested_optimizer": false
# }
# EOT
# export PL_DEEPSPEED_CONFIG_PATH=$config_json

TRAINER_ARGS="
    --max_epochs 20 \
    --gpus $NUM_GPUS \
    --log_every_n_steps 1 \
    --precision 16 \
    --save_dir $ROOT_DIR/commonsenseqa_outputs \
    --save_top_k 3 \
    --monitor val_accuracy_epoch \
    --mode max \
    --timestamp $TIMESTAMP \
    --gradient_clip_val 1.0 \
    --strategy ddp \
    --num_nodes $NUM_NODES \
    --patience 5 \
    --train \
"
    # 不用test，因为csqa的test没有答案，默认都选A
    # --test \
    # --monitor avg_val_loss \
    # --strategy deepspeed_stage_1 \
    # --val_check_interval 0.5 \
    # --check_val_every_n_epoch 1 \

DATA_DIR=/cognitive_comp/zhuxinyu/datasets/commonsenseqa
DATA_ARGS="
    --data_dir $DATA_DIR \
    --train_data train_rand_split.jsonl \
    --valid_data dev_rand_split.jsonl \
    --test_data test_rand_split_no_answers.jsonl \
    --num_workers 16 \
    --micro_batch_size 16 \
    --global_batch_size 32 \
    --valid_batch_size 32 \
    --test_batch_size 8 \
"
    # --recreate_dataset \
    # --source_max_token_len 512 \

MODEL_ARGS="
    --seed 19990303 \
    --model_type gpt \
    --model_name /cognitive_comp/zhuxinyu/pretrained_models/gpt2 \
    --lr 1e-5 \
    --l2 0. \
    --warmup 0.1 \
    --show_training_ex 100 \
    --scheduler linear \
"
    # --loss_on_prefix \
    # --model_name gpt-j-6B \
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/commonsenseqa_outputs/gpt2-CSQA-04-26_18-07/epoch=11-avg_val_loss=0.1507.ckpt \


SCRIPTS_PATH=$ROOT_DIR/gpt_training_csqa.py
# SCRIPTS_PATH=$ROOT_DIR/commonsenseqa_data_model.py

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
# srun  --nodes=$NUM_NODES --gres=gpu:$NUM_GPUS --cpus-per-gpu=32 bash -c 'python $CMD'
bash -c 'python $CMD'

# SINGULARITY_PATH=/cognitive_comp/zhuxinyu/container/pytorch21_06_py3_docker_image_v2.sif
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python $CMD'

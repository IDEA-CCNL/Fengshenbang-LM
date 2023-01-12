#!/bin/bash
#SBATCH --job-name=pretrain_wenzhong # create a short name for your job
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH -c 8 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -e %x-%j.err # output and error log file names (%x for job id)

MAX_LEN=2048
# large model
MODEL_NAME=Wenzhong2.0-GPT2-13B
RUN_NAME=gpt_v0_rope_large

# small model
MODEL_NAME=Wenzhong2.0-GPT2-110M-Test
RUN_NAME=gpt_v0_rope

# gau model
# MODEL_NAME=Wenzhong2.0-GAU-110M
# RUN_NAME=gpt_v0_gau_pretrain-110M

ROOT_DIR=/cognitive_comp/yangqi/logs/gpt_pretrain/$MAX_LEN/$RUN_NAME
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extensions
DATA_DIR=/cognitive_comp/common_data/dialogue/PretrainData


if [ ! -d ${ROOT_DIR} ];then
  mkdir ${ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=1

MICRO_BATCH_SIZE=4

CONFIG_JSON="${ROOT_DIR}/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=3


# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $CONFIG_JSON
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "gradient_clipping": 1,
    "zero_optimization":{
        "stage": 3,
        "offload_optimizer":{
            "device":"cpu",
            "pin_memory":true
        },
        "offload_param":{
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
                0.999
            ],
            "eps": 1e-08,
            "lr": 1e-06,
            "weight_decay": 1e-05
        },
        "type": "Adam"
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 1e-04,
            "warmup_min_lr": 1e-08,
            "total_num_steps": 10000,
            "warmup_num_steps" : 600
        },
        "type": "WarmupDecayLR"
    },
    "steps_per_print": 100,
    "zero_allow_untested_optimizer": false
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
export TORCH_EXTENSIONS_DIR=/cognitive_comp/yangqi/torch_extensions
### End
        # --train_file ${DATA_DIR}/train.json \
        # --val_file  ${DATA_DIR}/dev.json \
        # --test_file  ${DATA_DIR}/test.json \--datasets_name wudao_180g_test\
DATA_ARGS="\
        --datasets_name wudao_180g_10k \
        --dataloader_workers 4 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        "


MODEL_ARGS="\
        --model_path /cognitive_comp/yangqi/model/${MODEL_NAME} \
        --learning_rate 1e-4 \
        --weight_decay 1e-5 \
        --warmup_ratio 0.01 \
        --sample_content_key text \
        --max_seq_length $MAX_LEN \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${ROOT_DIR}/ckpt \
        --load_ckpt_path ${ROOT_DIR}/ckpt/last.ckpt \
        --every_n_train_steps 10 \
        "
#         --val_check_interval 0.1 \
TRAINER_ARGS="\
        --max_epoch 5 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_3 \
        --log_every_n_steps 10 \
        --precision 16 \
        --default_root_dir ${ROOT_DIR} \
        --tensorboard_dir ${ROOT_DIR} \
        --replace_sampler_ddp true \
        --val_check_interval 0.1 \
        --gradient_clip_val 1.0 \
        --limit_train_batches 1.0 \
        --limit_val_batches 0.5 \
        --limit_test_batches 0.5 \
        "

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

CUDA_VISIBLE_DEVICES=1 python3 pretrain_gpt_dev.py $options > ${ROOT_DIR}/test.log
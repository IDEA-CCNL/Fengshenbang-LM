#!/bin/bash
#SBATCH --job-name=wenzhong_research # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=8 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
set -x -e

MODEL_NAME=Wenzhong-GPT2-110M
RUN_NAME=gpt_v0_nokno
ROOT_DIR=../../workspace/log/$RUN_NAME
DATA_DIR=../../workspace/data/$RUN_NAME

if [ ! -d ${ROOT_DIR} ];then
  mkdir ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

config_json="$ROOT_DIR/$MODEL_NAME.ds_config.json"
export MASTER_PORT=$[RANDOM%10000+40000]

MICRO_BATCH_SIZE=16

cat <<EOT > $config_json
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "gradient_clipping": 1,
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
                0.999
            ],
            "eps": 1e-08,
            "lr": 1e-05,
            "weight_decay": 1e-05
        },
        "type": "Adam"
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 1e-04,
            "warmup_min_lr": 5e-06,
            "total_num_steps": 5000,
            "warmup_num_steps" : 200
        },
        "type": "WarmupDecayLR"
    },
    "steps_per_print": 100,
    "zero_allow_untested_optimizer": false
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=../../workspace/torch_extensions


DATA_ARGS=" \
        --train_file $DATA_DIR/train.json \
        --val_file $DATA_DIR/dev.json \
        --test_file $DATA_DIR/test.json \
        --num_workers 8 \
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --max_seq_lengt 512 \
        --max_src_length 128 \
        --max_kno_length 256 \
        --max_tgt_length 128 \
        "  

MODEL_ARGS="\
        --model_path ../workspace/model/$MODEL_NAME \
        --tokenizer_path ../workspace/model/$MODEL_NAME \
        --learning_rate 1e-4 \
        --min_learning_rate 1e-5 \
        --lr_decay_steps 10000 \
        --weight_decay 1e-2 \
        --warmup_steps 1000 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 3 \
        --mode min \
        --save_last \
        --every_n_train_steps 2000 \
        --save_ckpt_path $ROOT_DIR/ckpt/ \
        --load_ckpt_path $ROOT_DIR/ckpt/ \
        --filename model-{step:02d}-{train_loss:.4f} \
        "

TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 3 \
        --gpus 1 \
        --num_nodes 1 \
        --collator dial \
        --strategy deepspeed_stage_1 \
        --log_every_n_steps 100 \
        --val_check_interval 0.5 \
        --accumulate_grad_batches 1 \
        --default_root_dir $ROOT_DIR \
        --tensorboard_dir $ROOT_DIR \
    "        
#--label_smooth 0.1 \
# --resume_from_checkpoint $ROOT_DIR/ckpt/last.ckpt \
export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

export SCRIPT_PATH=./finetune_multitask.py
CUDA_VISIBLE_DEVICES=3 python3 ${SCRIPT_PATH} $options  > $ROOT_DIR/test.log

# CKPT_NAME=last.ckpt
# PRED_FILE=dev_pred.json
# QA_TYPE=3

# pred
# TRAINER_ARGS="\
#         --gradient_clip_val 1.0 \
#         --max_epochs 10 \
#         --gpus 1 \
#         --num_nodes 1 \
#         --strategy deepspeed_stage_1 \
#         --log_every_n_steps 10 \
#         --val_check_interval 0.5 \
#         --accumulate_grad_batches 1 \
#         --default_root_dir $ROOT_DIR \
#         --tensorboard_dir $ROOT_DIR \
#         --label_smooth 0.1 \
#         --do_eval_only \
#         --pred_file  $PRED_FILE \
#         --test_model $CKPT_NAME \
#         --qa_type $QA_TYPE \
#         --sample_num 128 \
#     "

# export options=" \
#         $DATA_ARGS \
#         $MODEL_ARGS \
#         $MODEL_CHECKPOINT_ARGS \
#         $TRAINER_ARGS \
#         "
# export SCRIPT_PATH=./finetune_multitask.py
# # CUDA_VISIBLE_DEVICES=0 
# python3 ${SCRIPT_PATH} $options > $ROOT_DIR/test.log




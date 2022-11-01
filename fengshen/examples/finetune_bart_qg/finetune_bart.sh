#!/bin/bash
#SBATCH --job-name=bart_qg # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=10 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
set -x -e

MODEL_NAME=IDEA-CCNL/Randeng-BART-139M
RUN_NAME=bart_v0_test
ROOT_DIR=../../workspace/log/$RUN_NAME

config_json="$ROOT_DIR/$MODEL_NAME.ds_config.json"
export MASTER_PORT=$[RANDOM%10000+40000]

MICRO_BATCH_SIZE=32

cat <<EOT > $config_json
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "gradient_clipping": 1,
    "zero_optimization": {
        "stage": 1
    },
    "fp16": {
        "enabled": true,
    }
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=../../workspace/torch_extensions

DATA_ARGS=" \
        --train_file train.json \
        --val_file dev.json \
        --test_file test.json \
        --tokenizer_type bart \
        --num_workers 8 \
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --max_seq_lengt 512 \
        --max_src_length 32 \
        --max_kno_length 416 \
        --max_tgt_length 64 \
        --mask_ans_style anstoken_multispan \
        "  

MODEL_ARGS="\
        --model_path $MODEL_NAME/ \
        --learning_rate 1e-4 \
        --min_learning_rate 1e-8 \
        --lr_decay_steps 100000 \
        --weight_decay 1e-2 \
        --warmup_steps 1000 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_loss \
        --save_top_k 3 \
        --mode min \
        --save_last \
        --every_n_train_steps 5000 \
        --save_ckpt_path $ROOT_DIR/ckpt/ \
        --load_ckpt_path $ROOT_DIR/ckpt/ \
        --filename model-{step:02d}-{train_loss:.4f} \
        "

TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 1 \
        --gpus 1 \
        --num_nodes 1 \
        --strategy ddp \
        --log_every_n_steps 100 \
        --val_check_interval 0.5 \
        --accumulate_grad_batches 1 \
        --default_root_dir $ROOT_DIR \
        --tensorboard_dir $ROOT_DIR \
        --label_smooth 0.1 \
    "



export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "
# test
export SCRIPT_PATH=./finetune_bart.py

python3 ${SCRIPT_PATH} $options > $ROOT_DIR/test.log


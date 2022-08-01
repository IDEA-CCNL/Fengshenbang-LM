#!/bin/bash
#SBATCH --job-name=pretrain_bart # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050


MODEL_NAME=bert-1.3B

config_json="./$MODEL_NAME.ds_config.json"
((MASTER_PORT=$RANDOM%10000+40000))
echo $MASTER_PORT
ZERO_STAGE=2
MICRO_BATCH_SIZE=16

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "allgather_bucket_size": 2e8
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
            "lr": 1e-04,
            "weight_decay": 0.01
        },
        "type": "Adam"
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 1e-04,
            "warmup_min_lr": 1e-05,
            "total_num_steps": 536877,
            "warmup_num_steps" : 50000
        },
        "type": "WarmupDecayLR"
    },
    "steps_per_print": 100,
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "zero_allow_untested_optimizer": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/home/wuziwei/torch_extendsions

DATA_ARGS="\
        --datasets_name wudao_180g \
        --num_workers 16 \
        --train_batchsize $MICRO_BATCH_SIZE 
        "

MODEL_ARGS="\
        --model_path /data0/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_bert/wudao180g_bert_base \
        --learning_rate 1e-5 \
        --weight_decay 0.01 \
        --warmup 0.001 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 3 \
        --mode min \
        --save_last \
        --every_n_train_steps 5000 \
        --dirpath /data0/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_bert/$MODEL_NAME \
        --filename model-{step:02d}-{train_loss:.4f} \
        "
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 1 \
        --gpus 2 \
        --num_nodes 1 \
        --strategy ddp \
        --log_every_n_steps 100 \
        --val_check_interval 0.1 \
        --check_val_every_n_epoch 1 \
        --accumulate_grad_batches 1 \
        --resume_from_checkpoint /data0/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_bert/$MODEL_NAME/last.ckpt \
        --default_root_dir /data0/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_bert/$MODEL_NAME \
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

export SCRIPT_PATH=/data0/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_bert/pretrain_bert.py

bash -c 'python3 $SCRIPT_PATH $options'


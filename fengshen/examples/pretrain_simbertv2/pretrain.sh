#!/bin/bash
#SBATCH --job-name=pretrain_bert # create a short name for your job
#SBATCH --nodes=2 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050


MODEL_NAME=bert-sim

config_json="./$MODEL_NAME.ds_config.json"
((MASTER_PORT=$RANDOM%10000+40000))
echo $MASTER_PORT
ZERO_STAGE=2
MICRO_BATCH_SIZE=8

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
            "total_num_steps": 399378,
            "warmup_num_steps" : 10000
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
export TORCH_EXTENSIONS_DIR=/raid/wuziwei/torch_extendsions

DATA_ARGS="\
        --datasets_name wudao_180g \
        --num_workers 16 \
        --train_batchsize $MICRO_BATCH_SIZE 
        "

MODEL_ARGS="\
        --model_path /raid/wuziwei/pretrained_model_hf/bert_base4wudao \
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
        --dirpath /raid/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_simbertv2/$MODEL_NAME \
        --filename model-{step:02d}-{train_loss:.4f} \
        "
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 10 \
        --gpus 2 \
        --num_nodes 1 \
        --strategy ddp \
        --log_every_n_steps 1000 \
        --val_check_interval 1000 \
        --check_val_every_n_epoch 1 \
        --accumulate_grad_batches 1 \
        --resume_from_checkpoint /raid/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_simbertv2/$MODEL_NAME/last.ckpt \
        --default_root_dir /raid/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_simbertv2/$MODEL_NAME \
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

export SCRIPT_PATH=/raid/wuziwei/codes/Fengshenbang-LM/fengshen/examples/pretrain_simbertv2/pretrain.py

bash -c 'python3 $SCRIPT_PATH $options'


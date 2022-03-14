#!/bin/bash
#SBATCH --job-name=els_afqmc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --ntasks-per-node=1          
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err

set -x -e

export MASTER_PORT=$[RANDOM%10000+40000]

export TORCH_EXTENSIONS_DIR=/cognitive_comp/gaoxinyu/cache/torch_extendsions

TASK=afqmc
TEXTA_NAME=sentence1
TEXTB_NAME=sentence2
LABEL_NAME=label
ID_NAME=id

BATCH_SIZE=10
VAL_BATCH_SIZE=32

DATA_DIR=/cognitive_comp/yangping/data/ChineseCLUE_DATA/${TASK}_public/
PRETRAINED_MODEL_PATH='IDEA-CCNL/Erlangshen-1.3B'

CHECKPOINT_PATH=/cognitive_comp/gaoxinyu/finetune_model/$TASK/
DEFAULT_ROOT_DIR=/cognitive_comp/gaoxinyu/finetune_model/
OUTPUT_PATH=/cognitive_comp/gaoxinyu/finetune_model/${TASK}_predict.json

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 500000000,
        "allgather_bucket_size": 1000000000
    },
    "activation_checkpointing": {
        "contiguous_memory_optimization": false,
        "partition_activations": false
    },
    "fp16": {
        "enabled": true,
        "hysteresis": 2,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "min_loss_scale": 1
    },
    "gradient_clipping": 1,
    "optimizer": {
        "params": {
            "betas": [
                0.9,
                0.95
            ],
            "eps": 1e-08,
            "lr": 1e-05,
            "weight_decay": 0.01
        },
        "type": "Adam"
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 1e-05,
            "warmup_min_lr": 5e-06
        },
        "type": "WarmupLR"
    },
    "steps_per_print": 100,
    "train_micro_batch_size_per_gpu": 10,
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": false,
    "gradient_accumulation_steps": 1,
    "train_batch_size": 10
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json


DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.json \
        --valid_data dev.json \
        --test_data test.json \
        --train_batchsize $BATCH_SIZE \
        --valid_batchsize $VAL_BATCH_SIZE \
        --max_length 512 \
        --texta_name $TEXTA_NAME \
        --textb_name $TEXTB_NAME \
        --label_name $LABEL_NAME \
        --id_name $ID_NAME \
        "

MODEL_ARGS="\
        --learning_rate 5e-5 \
        --weight_decay 1e-1 \
        --warmup 0.065 \
        --num_labels 2 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_acc \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 1000 \
        --save_weights_only True \
        --dirpath $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_acc:.4f} \
        "


TRAINER_ARGS="\
        --max_epochs 5 \
        --gpus 1 \
        --strategy deepspeed_stage_1 \
        --gradient_clip_val 1.0 \
        --check_val_every_n_epoch 1 \
        --val_check_interval 1.0 \
        --default_root_dir $DEFAULT_ROOT_DIR \
        "

options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --output_save_path $OUTPUT_PATH \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
SCRIPT_PATH=../../examples/finetune_classification.py

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 2 \
    --nnodes 1 \
    --master_addr 127.0.0.1 \
    --master_port $MASTER_PORT \
    --node_rank 0 \
    --max_restarts 0 \
    "


#CUDA_VISIBLE_DEVICES=7 singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH python3 $SCRIPT_PATH $options
singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH python3 $SCRIPT_PATH $options
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c "export PATH="$PATH:/home/gaoxinyu/.local/bin/";deepspeed --autotuning tune  $SCRIPT_PATH --deepspeed $PL_DEEPSPEED_CONFIG_PATH $options;"
# if you don't have docker, you can use the following command
# python3 $SCRIPT_PATH $options


#!/bin/bash
#SBATCH --job-name=finetune_wenzhong
#SBATCH --cpus-per-task=50
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err

set -x -e

export MASTER_PORT=$[RANDOM%10000+50000]
export TORCH_EXTENSIONS_DIR=/cognitive_comp/gaoxinyu/torch_extendsions

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=1
ROOT_DIR=/cognitive_comp/gaoxinyu/FS/fengshen/fengshen

ZERO_STAGE=3

config_json="$ROOT_DIR/ds_config.$SLURM_JOBID.json"
#config_json="$ROOT_DIR/ds_config.wzw.json"
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
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
    "optimizer":{
        "type":"Adam",
        "params":{
            "lr": 1e-5,
            "weight_decay":0.01
        }
    },
    "scheduler":{
        "type":"WarmupLR",
        "params":{
            "warmup_min_lr":5e-6,
            "warmup_max_lr":1e-5
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
    --max_epochs 2 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy deepspeed_stage_3 \
    --precision 16 \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --monitor train_loss \
    --mode min \
    --save_last \
"
DATA_DIR=/cognitive_comp/gaoxinyu/data/yuyuan
DATA_ARGS="
    --data_dir $DATA_DIR \
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --train_data train.txt \
    --valid_data valid.txt \
    --test_data  test.txt
"

MODEL_ARGS="
    --pretrained_model_path /cognitive_comp/gaoxinyu/hf_model/wenzhong \
    --output_save_path $ROOT_DIR/predict.json \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup 0.01 \
"

SCRIPTS_PATH=/cognitive_comp/gaoxinyu/FS/fengshen/finetune_wenzhong.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"

clear; srun --jobid $SLURM_JOBID singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python $CMD'
# bash -c 'python $CMD'
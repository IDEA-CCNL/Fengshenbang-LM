#!/bin/bash
#SBATCH --job-name=mt5_large_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4               # number of gpus
#SBATCH -o /cognitive_comp/ganruyi/fengshen/mt5_large_summary/%x-%j.log
#SBATCH -e /cognitive_comp/ganruyi/fengshen/mt5_large_summary/%x-%j.err

set -x -e

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=16
ROOT_DIR=/cognitive_comp/ganruyi/fengshen/mt5_large_summary

ZERO_STAGE=2

config_json="$ROOT_DIR/ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": 16,
  "steps_per_print": 100,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "contiguous_gradients": false,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 500000000
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-5,
      "betas": [
        0.9,
        0.95
      ],
      "eps": 1e-8,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params":{
      "warmup_min_lr": 5e-6,
      "warmup_max_lr": 1e-5
    }
  },
  "zero_allow_untested_optimizer": false,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": false
}
EOT

# export PL_DEEPSPEED_CONFIG_PATH=$config_json

TRAINER_ARGS="
    --max_epochs 2 \
    --gpus 4 \
    --num_nodes 1 \
    --strategy ddp \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --monitor train_loss \
    --mode min \
    --save_last \
"
DATA_DIR=/cognitive_comp/ganruyi/data_datasets_LCSTS_LCSTS/
prompt="summary:"
DATA_ARGS="
    --data_dir $DATA_DIR
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --train_data train.jsonl\
    --valid_data valid.jsonl\
    --test_data  valid.jsonl\
    --prompt $prompt \
"

MODEL_ARGS="
    --pretrained_model_path /cognitive_comp/ganruyi/hf_models/google/mt5-large \
    --output_save_path $ROOT_DIR/mt5_large_predict_lcsts.json \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup 0.01 \
"

SCRIPTS_PATH=/cognitive_comp/ganruyi/fengshen/examples/mt5_summary.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD

SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif
#singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH python $CMD

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
clear; srun singularity exec --nv -B /cognitive_comp/ganruyi/:/cognitive_comp/ganruyi/ $SINGULARITY_PATH bash -c 'python $CMD'
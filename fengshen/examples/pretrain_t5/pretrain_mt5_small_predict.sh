#!/bin/bash
#SBATCH --job-name=t5_cn_small_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o /cognitive_comp/ganruyi/fengshen/t5_cn_small_pretrain/%x-%j.log
#SBATCH -e /cognitive_comp/ganruyi/fengshen/t5_cn_small_pretrain/%x-%j.err

set -x -e

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=128
ROOT_DIR=/cognitive_comp/ganruyi/fengshen/t5_cn_small_pretrain/

ZERO_STAGE=2

config_json="$ROOT_DIR/ds_config.t5_cn_small_pretrain.json"
export MASTER_PORT=$[RANDOM%10000+30000]
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": 128,
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
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
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
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 10000
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

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions
# strategy=ddp
strategy=deepspeed_stage_2

TRAINER_ARGS="
    --max_epochs 1 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy ${strategy} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 10 \
    --monitor train_loss \
    --mode min \
    --save_last \
    --val_check_interval 0.01 \
    --accumulate_grad_batches 8 \
    --resume_from_checkpoint /cognitive_comp/ganruyi/fengshen/t5_cn_small_pretrain/old-ckpt/last.ckpt \
    --do_eval_only \
"
# --accumulate_grad_batches 8 \
DATA_DIR=wudao_180g_mt5_tokenized

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --train_data wudao_180g_mt5_tokenized\
    --train_split_size 0.999 \
    --max_seq_length 1024 \
"

MODEL_ARGS="
    --pretrained_model_path /cognitive_comp/ganruyi/hf_models/google/mt5-small \
    --new_vocab_path /cognitive_comp/ganruyi/hf_models/t5_cn_small/sentencepiece_cn.model \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --keep_tokens_path /cognitive_comp/ganruyi/hf_models/t5_cn_small/sentencepiece_cn_keep_tokens.json \
"

SCRIPTS_PATH=/cognitive_comp/ganruyi/fengshen/pretrain_t5.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD

# SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
# clear; srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c '/home/ganruyi/anaconda3/bin/python $CMD'
/home/ganruyi/anaconda3/bin/python $CMD
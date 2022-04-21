#!/bin/bash
#SBATCH --job-name=medical_qa_finetune
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH -o /cognitive_comp/wuziwei/task/fs_medical_qa_finetune/%x-%j.log
#SBATCH -e /cognitive_comp/wuziwei/task/fs_medical_qa_finetune/%x-%j.err
#SBATCH -x dgx[050,049]

#export NCCL_DEBUG=INFO

# export PATH=$PATH:/cognitive_comp/wuziwei/codes/fengshen/fengshen
set -x -e

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=1
ROOT_DIR=/cognitive_comp/wuziwei/task/fs_medical_qa_finetune

ZERO_STAGE=2

config_json="$ROOT_DIR/training_config.json"
export MASTER_PORT=$[RANDOM%10000+30000]

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
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-5,
      "betas": [0.9,0.95],
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
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 32,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": false,
  "train_micro_batch_size_per_gpu": 1,
  "steps_per_print": 100,
  "gradient_clipping": 1.0
}
EOT

# export PL_DEEPSPEED_CONFIG_PATH=$config_json
export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/wuziwei/torch_extendsions
TRAINER_ARGS="
    --max_epochs 10 \
    --gpus 16 \
    --num_nodes 2 \
    --strategy deepspeed_stage_2 \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --monitor train_loss \
    --mode min \
    --save_last \
"
DATA_DIR=/cognitive_comp/wuziwei/task-data/medical_qa
DATA_ARGS="
    --data_dir $DATA_DIR \
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --train_data train.txt \
    --valid_data valid.txt \
    --test_data  test.txt
"

# PRETRAINED_MODEL_PATH=/cognitive_comp/wuziwei/pretrained_model_hf/gpt2
PRETRAINED_MODEL_PATH=/cognitive_comp/wuziwei/pretrained_model_hf/medical_v2
MODEL_ARGS="
    --pretrained_model_path ${PRETRAINED_MODEL_PATH} \
    --output_save_path $ROOT_DIR/predict.json \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup 0.01 \
"

SCRIPTS_PATH=/cognitive_comp/wuziwei/codes/fengshen/fengshen/examples/GPT_pretrain_finetune/finetune_medicalQA.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD

SINGULARITY_PATH=/cognitive_comp/wuziwei/container/oneflow-cuda11.sif
# singularity exec --nv -B /cognitive_comp/wuziwei/:/cognitive_comp/wuziwei/ $SINGULARITY_PATH python $CMD

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"

srun singularity exec --nv -B /cognitive_comp/wuziwei/:/cognitive_comp/wuziwei/ $SINGULARITY_PATH bash -c 'python $CMD'

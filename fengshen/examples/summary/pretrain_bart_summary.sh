#!/bin/bash
#SBATCH --job-name=bart_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4               # number of gpus
#SBATCH -o %x-%j.log

set -x -e

echo "START TIME: $(date)"
MODEL_NAME=bart-base
MICRO_BATCH_SIZE=16
ROOT_DIR=/cognitive_comp/dongxiaoqun/finetune/${MODEL_NAME}

ZERO_STAGE=1
export TORCH_EXTENSIONS_DIR=/cognitive_comp/dongxiaoqun/torch_extendsions
config_json="./ds_config.${MODEL_NAME}.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
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
      "lr": 1e-4,
      "betas": [
        0.9,
        0.95
      ],
      "eps": 1e-8,
      "weight_decay": 5e-2
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params":{
      "warmup_min_lr": 5e-6,
      "warmup_max_lr": 1e-4
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
    --gpus 1 \
    --num_nodes 1 \
    --strategy deepspeed_stage_${ZERO_STAGE} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --monitor val_loss \
    --mode min \
    --save_last \
    --every_n_train_steps 0 \
    --val_check_interval 0.1 \
"

prompt='"'
DATA_ARGS="
    --datasets_name lcsts \
    --num_workers 8 \
    --train_batchsize $MICRO_BATCH_SIZE \
    --val_batchsize $MICRO_BATCH_SIZE \
    --test_batchsize $MICRO_BATCH_SIZE \
    --max_enc_length 128 \
    --max_dec_length 64 \
    --val_datasets_field val \
    --prompt $prompt \
"

MODEL_ARGS="
    --pretrained_model_path /cognitive_comp/gaoxinyu/pretrained_model/bart-base \
    --output_save_path $ROOT_DIR/${MODEL_NAME}_predict_lcsts.json \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --precision 16 \
"

SCRIPTS_PATH=seq2seq_summary.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD

#singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH python $CMD

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
# srun --nodes=1 --gres=gpu:4 --ntasks-per-node=4 --cpus-per-gpu=20 
source activate
conda activate torchnew
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=30 -o ${MODEL_NAME}-%J.log --jobid=229623 bash -c 'python3 $SCRIPT_PATH $CMD'

#!/bin/bash
#SBATCH --job-name=cbart_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err

set -x -e

echo "START TIME: $(date)"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#MASTER_ADDR=127.0.0.1
MASTER_PORT=$[RANDOM%10000+50000]

GPUS_PER_NODE=1
NNODES=1   # switch to 128
TP_SIZE=1    # always fixed to the size of a single node
PP_SIZE=1    # NLAYERS must be a multiple of PP_SIZE here

MICRO_BATCH_SIZE=16

ZERO_STAGE=2

config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_accumulation_steps": 1,
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
      "warmup_max_lr": 1e-5,
      "warmup_num_steps": "auto"
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


CBART_ARGS="
    --model_path /cognitive_comp/gaoxinyu/FS/cbart/checkpoints/cpt \
    --dataset wudao \
    --num_labels 5 \
"


TRAINER_ARGS="
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --per_device_eval_batch_size $MICRO_BATCH_SIZE \
    --output_dir /cognitive_comp/gaoxinyu/hf_model/cbart/ \
    --logging_dir /cognitive_comp/gaoxinyu/hf_model/cbart-tensorboard/ \
    --logging_steps 100 \
    --eval_steps 100 \
    --save_steps 20000 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --evaluation_strategy steps \
    --max_grad_norm 1.0 \
    --fp16 True \
    --adam_beta2 0.95 \
"

DEEPSPEED_ARGS=" \
    --deepspeed ${config_json} \
    "

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    "

SCRIPTS_PATH=/cognitive_comp/gaoxinyu/FS/Fengshenbang-LM

export CMD=" \
    $SCRIPTS_PATH/pretrain_cbart.py \
    $TRAINER_ARGS \
    $DEEPSPEED_ARGS \
    $CBART_ARGS \
    "

echo $CMD

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
#singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH $LAUNCHER --node_rank 0 $CMD

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
clear; srun --jobid $SLURM_JOBID singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
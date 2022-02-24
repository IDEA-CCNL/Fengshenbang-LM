#!/bin/bash
#SBATCH --job-name=cbart_pretrain_lightning
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2               # number of gpus
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err

set -x -e

echo "START TIME: $(date)"


MICRO_BATCH_SIZE=16

ZERO_STAGE=2

config_json="./ds_config.$SLURM_JOBID.json"

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

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TOKENIZERS_PARALLELISM=True

TRAINER_ARGS="
    --max_epochs 1 \
    --gpus 2 \
    --num_nodes 2 \
    --strategy deepspeed_stage_2 \
    --precision 16 \
    --default_root_dir /cognitive_comp/gaoxinyu/ln_model/cbart_lightning_example \
"

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --eval_batchsize $MICRO_BATCH_SIZE \
    --test_batchsize $MICRO_BATCH_SIZE \
    --train_datas /cognitive_comp/gaoxinyu/data/wudao-small/train_synthetic_max_insert_label3_insert_mode0_0 \
    --valid_datas /cognitive_comp/gaoxinyu/data/wudao-small/dev_synthetic_max_insert_label3_insert_mode0_0 \
    --test_datas /cognitive_comp/gaoxinyu/data/wudao-small/dev_synthetic_max_insert_label3_insert_mode0_0 \
    --input_tensor_name target_ids_list label_ids_list incorrect_input_ids_list \
"

CBART_ARGS="
    --model_path /cognitive_comp/gaoxinyu/FS/cbart/checkpoints/cpt \
    --num_labels 5 \
    --label_weights 0.00228146 0.0091339  0.07943265 0.12724317 0.78190883
"

DEEPSPEED_ARGS=" \
    --deepspeed ${config_json} \
    "

SCRIPTS_PATH=/cognitive_comp/gaoxinyu/FS/Fengshenbang-LM

export CMD=" \
    $SCRIPTS_PATH/pretrain_cbart_lightning.py \
    $TRAINER_ARGS \
    $CBART_ARGS \
    $DATA_ARGS \
    "

echo $CMD

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
#singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH python $CMD

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
clear; srun --jobid $SLURM_JOBID singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH bash -c 'python $CMD'
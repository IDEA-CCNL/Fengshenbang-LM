#!/bin/bash
#SBATCH --job-name=randeng_pegasus_523M_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=30
#SBATCH -o %x-%j.log

set -x -e

echo "START TIME: $(date)"
MODEL_NAME=randeng_pegasus_523M_summary_last
MICRO_BATCH_SIZE=128
ROOT_DIR=/cognitive_comp/dongxiaoqun/finetune/${MODEL_NAME}

if [ ! -d ${ROOT_DIR} ];then
  mkdir ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

output_save_path=$ROOT_DIR/${MODEL_NAME}.json
if [ -f ${output_save_path} ];then
  echo ${output_save_path} exist, rm it!!!!!!!!!!!!!!!!!
  rm ${output_save_path}
fi

ZERO_STAGE=1

config_json="${ROOT_DIR}/ds_config.${MODEL_NAME}.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 1000,
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
      "lr": 5e-5,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "params": {
      "warmup_min_lr": 1e-8,
      "warmup_max_lr": 1e-4,
      "total_num_steps": 60000,
      "warmup_num_steps" : 1000
    },
    "type": "WarmupDecayLR"  
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
export TORCH_EXTENSIONS_DIR=/cognitive_comp/dongxiaoqun/torch_extendsions
# export MASTER_PORT=$[RANDOM%10000+50000]
# 
# --strategy deepspeed_stage_${ZERO_STAGE} \
TRAINER_ARGS="
    --max_epochs 10 \
    --gpus 8 \
    --num_nodes 1 \
    --strategy deepspeed_stage_${ZERO_STAGE} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --monitor val_loss \
    --mode min \
    --save_last \
    --every_n_train_steps 10000 \
    --val_check_interval 0.1 \
"

DATA_ARGS="
    --datasets_name lcsts \
    --num_workers 30 \
    --train_batchsize $MICRO_BATCH_SIZE \
    --val_batchsize $MICRO_BATCH_SIZE \
    --test_batchsize $MICRO_BATCH_SIZE \
    --max_seq_length 128 \
    --max_dec_length 64 \
"
# --prompt $prompt \
# --pretrained_model_path /cognitive_comp/ganruyi/experiments/randeng_t5_77M_summary/ckpt/hf_pretrained_epoch1_step75019 \

# mode_path="/cognitive_comp/dongxiaoqun/train_model/fengshen-pegasus-base/ckpt/hf_pretrained_epoch0_step22200/"
mode_path="/cognitive_comp/dongxiaoqun/train_model/fengshen-pegasus-large/ckpt/hf_pretrained_epoch0_step122000"
cp /cognitive_comp/dongxiaoqun/pretrained_model/pegasus-large/vocab.txt $mode_path/ 
MODEL_ARGS="
    --pretrained_model_path  $mode_path \
    --output_save_path $output_save_path \
"

SCRIPTS_PATH=/cognitive_comp/dongxiaoqun/project/Fengshenbang-LM/fengshen/examples/summary/finetune_pegasus_summary.py
SINGULARITY_PATH=/cognitive_comp/dongxiaoqun/software/docker/pytorch21_06_py3_docker_image_v2.sif

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python3 $CMD'


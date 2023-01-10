#!/bin/bash

#SBATCH --job-name=mbart_en_zh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8              # number of gpus
#SBATCH --cpus-per-task=32
#SBATCH -o %x-%j.log

set -x -e

echo "START TIME: $(date)"

MODEL_NAME=deltalm_en_zh
MICRO_BATCH_SIZE=16
ROOT_DIR=../../workspace
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}


if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
  echo ${MODEL_ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${MODEL_ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

output_save_path=${MODEL_ROOT_DIR}.json
if [ -f ${output_save_path} ];then
  echo ${output_save_path} exist, rm it!!!!!!!!!!!!!!!!!
  rm ${output_save_path}
fi

ZERO_STAGE=1

config_json="${MODEL_ROOT_DIR}/ds_config.${MODEL_NAME}.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 1000,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "contiguous_gradients": false
  },
  "zero_allow_untested_optimizer": false,
  "fp16": {
    "enabled": true
  },
  "wall_clock_breakdown": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json


TRAINER_ARGS="
    --max_epochs 20 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy deepspeed_stage_${ZERO_STAGE} \
    --default_root_dir ${MODEL_ROOT_DIR} \
    --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
    --save_top_k 3 \
    --monitor valid_sacrebleu \
    --mode max \
    --save_last \
    --every_n_train_steps 0 \
    --val_check_interval 0.2 \
    --label_smoothing 0.1 \
    --warmup_steps 4000 \
    --learning_rate 1e-7 \
    --adam_beta2 0.98 \
    --scheduler_type inverse_sqrt \
    --reverse_src_tgt \
    --tgt_zh \
"

DATA_ARGS="
    --datasets_name case_test \
    --num_workers 8 \
    --train_batchsize $MICRO_BATCH_SIZE \
    --val_batchsize $MICRO_BATCH_SIZE \
    --test_batchsize $MICRO_BATCH_SIZE \
    --val_datasets_field val \
    --max_enc_length 256 \
    --max_dec_length 256 \
"

mode_path="IDEA-CCNL/Randeng-Deltalm-362M-En-Zn"


MODEL_ARGS="
    --model_path  $mode_path \
    --output_save_path $output_save_path \
"

SCRIPTS_PATH=finetune_deltalm.py

cat $SCRIPTS_PATH

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD

source activate 
conda activate fengshen
# srun python3 $CMD
python3 $CMD

#!/bin/bash


###
 # @Author: dongxiaoqun
 # @Date: 2022-07-29 10:33:12
 # @LastEditors: dongxiaoqun
 # @LastEditTime: 2022-08-12 16:15:36
 # @FilePath: /dongxiaoqun/project/idea-ccnl/Fengshenbang-LM/fengshen/examples/translate/finetune_deltalm.sh
 # @Description: 
 # 
 # Copyright (c) 2022 by Idea, All Rights Reserved. 
### 

#SBATCH --job-name=randeng_pegasus_523M_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=32
#SBATCH -o %x-%j.log

set -x -e

echo "START TIME: $(date)"
# MODEL_NAME=opus_mt_de_en_test_other
MODEL_NAME=deltalm_base_de_en_smoothing
MICRO_BATCH_SIZE=16
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

TRAINER_ARGS="
    --max_epochs 50 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy deepspeed_stage_${ZERO_STAGE} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --monitor val_loss \
    --mode min \
    --save_last \
    --every_n_train_steps 5000 \
    --val_check_interval 1.0 \
    --label_smoothing 0 \
    --warmup_steps 4000 \
    --learning_rate 1e-7 \
    --scheduler_type inverse_sqrt \
"


DATA_ARGS="
    --datasets_name iwslt14_de_en \
    --num_workers 8 \
    --train_batchsize $MICRO_BATCH_SIZE \
    --val_batchsize $MICRO_BATCH_SIZE \
    --test_batchsize $MICRO_BATCH_SIZE \
    --val_datasets_field val \
    --max_enc_length 512 \
    --max_dec_length 512 \
"

mode_path="/cognitive_comp/dongxiaoqun/pretrained_model/deltalm/"
# mode_path="Helsinki-NLP/opus-mt-zh-en"
# mode_path="facebook/mbart-large-50"

MODEL_ARGS="
    --model_path  $mode_path \
    --output_save_path $output_save_path \
"

SCRIPTS_PATH=/cognitive_comp/dongxiaoqun/project/idea-ccnl/Fengshenbang-LM/fengshen/examples/translate/finetune_deltalm.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD

ls -l `which sh`

source activate 
conda activate torchnew
nvcc -V
which python3
# pip list | grep torch
# export CUDA_HOME=/cognitive_comp/dongxiaoqun/software/anaconda3/envs/torchnew/
# export PATH=$PATH:/cognitive_comp/dongxiaoqun/software/anaconda3/envs/dgx/
# srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=30 -o ${MODEL_NAME}-%J.log --jobid=228110 bash -c 'python3 $CMD'
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:2 --cpus-per-task=30 -o ${MODEL_NAME}-%J.log --jobid=228299 bash -c 'python3 $CMD'

# python $CMD 

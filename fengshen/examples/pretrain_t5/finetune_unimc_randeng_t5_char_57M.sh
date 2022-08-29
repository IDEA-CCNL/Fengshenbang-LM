#!/bin/bash
#SBATCH --job-name=finetune_unimc_randeng_t5_char_57M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=32 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err

set -x -e

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=64
# ROOT_DIR=/cognitive_comp/ganruyi/experiments/finetune_unimc_randeng_t5_char_57M/
SCRIPT_DIR=$(pwd)
ROOT_DIR=${SCRIPT_DIR}/../../../../experiments/finetune_unimc_randeng_t5_char_57M/
if [ ! -d ${ROOT_DIR} ];then
  mkdir ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

ZERO_STAGE=1

config_json="deepspeed_config.json"
export MASTER_PORT=$[RANDOM%10000+30000]
export CUDA_VISIBLE_DEVICES='2'
export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=${SCRIPT_DIR}/../../../../tmp/torch_extendsions
export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions
# strategy=ddp
strategy=deepspeed_stage_1

TRAINER_ARGS="
    --max_epochs 7 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy ${strategy} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --every_n_train_steps 10000 \
    --monitor val_loss \
    --mode min \
    --save_last \
    --val_check_interval 0.5 \
    --dataset_num_workers 4 \
    --dataloader_num_workers 4 \
    --replace_sampler_ddp False \
"
# --accumulate_grad_batches 8 \
# TRAIN_DATA_DIR=/cognitive_comp/yangping/data/unidata/multiplechoice/pretraining_alldata/alldata/train.json
# VALID_DATA_DIR=/cognitive_comp/yangping/data/unidata/multiplechoice/pretraining_alldata/alldata/dev.json
TRAIN_DATA_DIR=/cognitive_comp/yangping/data/unidata/multiplechoice/preprocessing_data/classification/tnews/train.json
VALID_DATA_DIR=/cognitive_comp/yangping/data/unidata/multiplechoice/preprocessing_data/classification/tnews/dev.json
DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --train_data_path ${VALID_DATA_DIR} \
    --valid_data_path ${VALID_DATA_DIR} \
    --max_seq_length 512 \
"

MODEL_ARGS="
    --pretrained_model_path /cognitive_comp/ganruyi/experiments/randeng_t5_char_57M/randeng_t5_char_57M \
    --tokenizer_type bert_tokenizer \
    --max_dec_length 16 \
"

SCRIPTS_PATH=${SCRIPT_DIR}/finetune_t5.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
~/anaconda3/bin/python $CMD
#!/bin/bash
#SBATCH --job-name=zen2_base_cmeee # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 
#SBATCH -o /cognitive_comp/lujunyu/experiments/ner_finetune/zen2_base_cmeee/%x-%j.log # output and error file name (%x=job name, %j=job id)
#SBATCH -p hgx


ROOT_DIR=../../workspace
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

MODEL_NAME=ner_bert_base
TASK=cmeee

MODEL_NAME=bert-base
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=1

MICRO_BATCH_SIZE=16

ZERO_STAGE=1
STRATEGY=deepspeed_stage_${ZERO_STAGE}

DATA_ARGS="\
        --num_workers 8 \
        --dataloader_workers 8 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        "

MODEL_ARGS="\
        --model_path $MODEL_ROOT_DIR/pretrain \
        --data_dir /cognitive_comp/lujunyu/data_zh/NER_Aligned/weibo \
        --model_type bert \
        --decode_type linear \
        --learning_rate 5e-5 \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_top_k -1 \
        --save_last \
        --every_n_train_steps 100 \
        --save_ckpt_path ${MODEL_ROOT_DIR} \
        "

TRAINER_ARGS="\
        --max_epochs 30 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --check_val_every_n_epoch 1 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
"

python3 finetune_sequence_tagging.py $options


# SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif
# python3 $SCRIPT_PATH $options
# source activate base
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH /home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options
# /home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options


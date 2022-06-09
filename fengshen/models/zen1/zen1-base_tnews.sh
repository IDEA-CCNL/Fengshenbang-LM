#!/bin/bash
#SBATCH --job-name=afqmc-bart-base # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=2 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 
#SBATCH -o %x-%j.log # output and error file name (%x=job name, %j=job id)

export CUDA_VISIBLE_DEVICES='6'
export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions

MODEL_NAME=fengshen-zen1

TASK=tnews
TEXTA_NAME=sentence
LABEL_NAME=label
ID_NAME=id


BATCH_SIZE=8
VAL_BATCH_SIZE=32
ZERO_STAGE=1
STRATEGY=deepspeed_stage_${ZERO_STAGE}

ROOT_DIR=/cognitive_comp/ganruyi/experiments/classification_finetune/${MODEL_NAME}_${TASK}
if [ ! -d ${ROOT_DIR} ];then
  mkdir -p ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

DATA_DIR=/cognitive_comp/yangping/data/ChineseCLUE_DATA/${TASK}_public/
# PRETRAINED_MODEL_PATH=/cognitive_comp/ganruyi/hf_models/zen/ZEN_pretrain_base_v0.1.0
PRETRAINED_MODEL_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/models/zen1/results/result-seqlevel-2022-06-08-14-52-23/checkpoint-16650

CHECKPOINT_PATH=${ROOT_DIR}/ckpt/
OUTPUT_PATH=${ROOT_DIR}/predict.json

#  --do_train \

python run_sequence_level_classification.py \
    --task_name $TASK \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --bert_model $PRETRAINED_MODEL_PATH \
    --max_seq_length 512 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 10

# SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif
# SCRIPT_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/examples/classification/finetune_classification.py

# python3 $SCRIPT_PATH $options
# source activate base
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH /home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options
# /home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options


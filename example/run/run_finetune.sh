#!/bin/bash
#SBATCH --job-name=slurm-test # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=6 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 

BERT_NAME=roberta_base
ROOT_PATH=cognitive_comp
TASK=afqmc
TRAIN_DATA_PATH=/$ROOT_PATH/yangping/data/ChineseCLUE_DATA/afqmc_public/train.json
DEV_DATA_PATH=/$ROOT_PATH/yangping/data/ChineseCLUE_DATA/afqmc_public/dev.json
TSET_DATA_PATH=/$ROOT_PATH/yangping/data/ChineseCLUE_DATA/afqmc_public/test.json

PRETRAINED_MODEL_PATH=/$ROOT_PATH/yangping/pretrained_model/$BERT_NAME/

CHECKPOINT_PATH=/$ROOT_PATH/yangping/checkpoints/modelevaluation/afqmc/model.pth
OUTPUT_PATH=/$ROOT_PATH/yangping/nlp/modelevaluation/output/${TASK}_predict.json
LOG_FILE_PATH=/$ROOT_PATH/yangping/nlp/modelevaluation/log/afqmc_$BERT_NAME.log



options=" \
       --train_data_path $TRAIN_DATA_PATH \
       --dev_data_path $DEV_DATA_PATH \
       --test_data_path $TSET_DATA_PATH \
       --checkpoints $CHECKPOINT_PATH \
       --pretrained_model_path $PRETRAINED_MODEL_PATH \
       --output_path $OUTPUT_PATH \
       --log_file_path $LOG_FILE_PATH \
       --batch_size 32 \
       --learning_rate 0.00002 \
       --max_length 64 \
       --epoch 17 \
       --model_type bert \
        "

DOCKER_PATH=/$ROOT_PATH/yangping/containers/pytorch21_06_py3_docker_image.sif
SCRIPT_PATH=/$ROOT_PATH/yangping/nlp/Fengshenbang-LM/example/finetune.py

nohup python3 $SCRIPT_PATH $options > afqmc_roberta_base.log 2>&1 & 

# python3 $SCRIPT_PATH $options
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options

# singularity exec --nv /$ROOT_PATH/yangping/containers/pytorch21_06_py3_docker_image.sif python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 27659 /$ROOT_PATH/yangping/nlp/finetune_tasks/main.py


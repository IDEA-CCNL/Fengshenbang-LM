#!/bin/bash
#SBATCH --job-name=slurm-test # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=2 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 

BERT_NAME=rolongformer_base
ROOT_PATH=cognitive_comp
TASK=csl

TRAIN_DATA_PATH=/$ROOT_PATH/yangping/data/ChineseCLUE_DATA/${TASK}_public/train.json
DEV_DATA_PATH=/$ROOT_PATH/yangping/data/ChineseCLUE_DATA/${TASK}_public/dev.json
TSET_DATA_PATH=/$ROOT_PATH/yangping/data/ChineseCLUE_DATA/${TASK}_public/test.json


PRETRAINED_MODEL_PATH=/$ROOT_PATH/yangping/pretrained_model/$BERT_NAME/

CHECKPOINT_PATH=/$ROOT_PATH/yangping/checkpoints/modelevaluation/csl/
OUTPUT_PATH=/$ROOT_PATH/yangping/nlp/modelevaluation/output/csl_predict.json

LOG_FILE_PATH=/$ROOT_PATH/yangping/nlp/modelevaluation/log/

options=" \
        --task_name $TASK \
        --train_data_path $TRAIN_DATA_PATH \
        --dev_data_path $DEV_DATA_PATH \
        --test_data_path $TSET_DATA_PATH \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --max_length 128 \
        --output_dir $LOG_FILE_PATH \
        --num_train_epochs 3 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 128 \
        --learning_rate 0.00002 \
        --warmup_steps 1000 \
        --weight_decay 0.01 \
        --max_grad_norm 0.1 \
        --logging_dir $LOG_FILE_PATH \
        --logging_steps 10 \
        --do_eval true \
        --evaluation_strategy steps \
        --save_steps 100 \
        --eval_steps 100 \
        --save_total_limit 3 \
        --load_best_model_at_end true \
        --metric_for_best_model eval_accuracy \
        "

DOCKER_PATH=/$ROOT_PATH/yangping/containers/pytorch21_06_py3_docker_image.sif
SCRIPT_PATH=/$ROOT_PATH/yangping/nlp/Fengshenbang-LM/fengshen/examples/finetune_classification.py

python3 $SCRIPT_PATH $options
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options


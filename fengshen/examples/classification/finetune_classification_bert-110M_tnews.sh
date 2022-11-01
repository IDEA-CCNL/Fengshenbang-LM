#!/bin/bash
#SBATCH --job-name=wzw-simcse
#SBATCH -p hgx
#SBATCH -N 1
#SBATCH -n 1 # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:1 # number of gpus
#SBATCH --cpus-per-task 4
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err




MODEL_TYPE=huggingface-bert
PRETRAINED_MODEL_PATH=/cognitive_comp/wuziwei/pretrained_model_hf/my-zh-simcse-bert-unsupAndsup-bak

ROOT_PATH=cognitive_comp
TASK=tnews

DATA_DIR=/$ROOT_PATH/yangping/data/ChineseCLUE_DATA/${TASK}_public/
CHECKPOINT_PATH=/cognitive_comp/wuziwei/finetune_checkpoint_dir/${TASK}/
OUTPUT_PATH=/cognitive_comp/wuziwei/finetune_logs/${TASK}/predict.json

DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.json \
        --valid_data dev.json \
        --test_data test1.0.json \
        --train_batchsize 32 \
        --valid_batchsize 128 \
        --max_length 128 \
        --texta_name sentence \
        --label_name label \
        --id_name id \
        "

MODEL_ARGS="\
        --learning_rate 0.00002 \
        --weight_decay 0.1 \
        --warmup 0.001 \
        --num_labels 15 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_acc \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 100 \
        --save_weights_only True \
        --dirpath $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_acc:.4f} \
        "

TRAINER_ARGS="\
        --max_epochs 7 \
        --gpus 1 \
        --check_val_every_n_epoch 1 \
        --val_check_interval 100 \
        --default_root_dir ./log/ \
        "


options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --output_save_path $OUTPUT_PATH \
        --model_type $MODEL_TYPE \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

# DOCKER_PATH=/$ROOT_PATH/yangping/containers/pytorch21_06_py3_docker_image.sif
SCRIPT_PATH=/cognitive_comp/wuziwei/codes/Fengshenbang-LM/fengshen/examples/classification/finetune_classification.py

# conda activate senEmb

python3 $SCRIPT_PATH $options
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options


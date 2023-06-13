#!/bin/bash
#SBATCH --job-name=slurm-test # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 

#SBATCH --requeue
#SBATCH --qos=preemptive


DATA_DIR=./data/cmrc2018  #数据集路径

PRETRAINED_MODEL_PATH=IDEA-CCNL/Erlangshen-Ubert-110M-Chinese

CHECKPOINT_PATH=./checkpoints

LOAD_CHECKPOINT_PATH=./checkpoints/last.ckpt

OUTPUT_PATH=./predict/cmrc2018_predict.json

DEFAULT_ROOT_DIR=./log


DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.json \
        --valid_data dev.json \
        --test_data dev.json \
        --batchsize 32 \
        --max_length 314 \
        "

MODEL_ARGS="\
        --learning_rate 0.00002 \
        --weight_decay 0.1 \
        --warmup 0.01 \
        --num_labels 1 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_span_acc \
        --save_top_k 5 \
        --mode max \
        --every_n_train_steps 100 \
        --save_weights_only true \
        --checkpoint_path $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_span_acc:.4f} \
        "

#--load_checkpoints_path $LOAD_CHECKPOINT_PATH \
TRAINER_ARGS="\
        --max_epochs 11 \
        --gpus 1 \
        --check_val_every_n_epoch 1 \
        --gradient_clip_val 0.25 \
        --val_check_interval 0.05 \
        --limit_val_batches 100 \
        --default_root_dir $DEFAULT_ROOT_DIR \
        "

options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --threshold 0.001 \
        --train \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

SCRIPT_PATH=./solution/clue_ubert.py
python3 $SCRIPT_PATH $options


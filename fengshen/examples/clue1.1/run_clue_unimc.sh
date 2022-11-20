#!/bin/bash
#SBATCH --job-name=slurm-test # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 

#SBATCH --requeue
#SBATCH --qos=preemptive

TASK=tnews #clue 上的任务 ，可选afqmc、tnews、iflytek、wsc、ocnli、csl、chid、c3
DATA_ROOT_PATH=./data  #数据集路径
DATA_DIR=$DATA_ROOT_PATH/$TASK

PRETRAINED_MODEL_PATH=IDEA-CCNL/Erlangshen-UniMC-RoBERTa-110M-Chinese  #预训练模型的路径

CHECKPOINT_PATH=./checkpoint  #训练模型保存的路径

LOAD_CHECKPOINT_PATH=./checkpoints/last.ckpt  #加载预训练好的模型

OUTPUT_PATH=./predict/${TASK}_predict.json

DEFAULT_ROOT_DIR=./log # 模型日志输出路径

DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.json \
        --valid_data dev.json \
        --test_data test1.1.json \
        --batchsize 1 \
        --max_length 512 \
        "

# 如果使用的是 UniMC-DeBERTa-1.4B模型，学习率要设置1e-6

MODEL_ARGS="\
        --learning_rate 0.000002 \
        --weight_decay 0.1 \
        --warmup 0.06 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_acc \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 100 \
        --save_ckpt_path $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_acc:.4f} \
        "

TRAINER_ARGS="\
        --max_epochs 17 \
        --gpus 1 \
        --check_val_every_n_epoch 1 \
        --val_check_interval 100 \
        --gradient_clip_val 0.25 \
        --default_root_dir $DEFAULT_ROOT_DIR \
        "

#--load_checkpoints_path $LOAD_CHECKPOINT_PATH \  如果想加载预训练好的ckpt模型，可以使用这个参数加载

options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --train \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

SCRIPT_PATH=./solution/clue_unimc.py
python3 $SCRIPT_PATH $options


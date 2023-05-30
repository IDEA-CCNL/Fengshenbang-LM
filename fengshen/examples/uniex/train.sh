#!/bin/bash
#SBATCH --job-name=eval_llama-7B # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=12 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pol-preempted # number of gpus per node


ROOT_PATH=cognitive_comp

DATA_DIR=/cognitive_comp/yangping/data/unidata/spandata/preprocessing_data/ner/cluener

PRETRAINED_MODEL_PATH=IDEA-CCNL/Erlangshen-UniEX-RoBERTa-110M-Chinese

CHECKPOINT_PATH=/$ROOT_PATH/yangping/checkpoints/mrc/uniex_test

DEFAULT_ROOT_DIR=/cognitive_comp/yangping/nlp/finetune/log/

DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.json \
        --valid_data dev.json \
        --test_data dev.json \
        --batchsize 16 \
        --max_length 512 \
        "


MODEL_ARGS="\
        --learning_rate 0.00001 \
        --weight_decay 0.1 \
        --warmup 0.1 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_f1 \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 40 \
        --save_weights_only true \
        --dirpath $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_f1:.4f} \
        "

#--load_checkpoints_path $LOAD_CHECKPOINT_PATH \
TRAINER_ARGS="\
        --max_epochs 47 \
        --gpus 1 \
        --check_val_every_n_epoch 1 \
        --gradient_clip_val 0.25 \
        --val_check_interval 40 \
        --default_root_dir $DEFAULT_ROOT_DIR \
        "

options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --output_path /cognitive_comp/yangping/nlp/finetune/scripts/UniEX/finetune/predict.json \
        --train \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

DOCKER_PATH=/$ROOT_PATH/yangping/containers/pytorch21_06_py3_docker_image.sif
SCRIPT_PATH=/cognitive_comp/yangping/nlp/Fengshenbang-LM/fengshen/examples/uniex/example.py

python3 $SCRIPT_PATH $options
# srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options


#!/bin/bash
#SBATCH --job-name=slurm-test # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=2 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 



MODEL_TYPE=fengshen-roformer
PRETRAINED_MODEL_PATH=IDEA-CCNL/Zhouwenwang-110M

ROOT_PATH=cognitive_comp
TASK=tnews

DATA_DIR=/$ROOT_PATH/yangping/data/ChineseCLUE_DATA/${TASK}_public/
CHECKPOINT_PATH=/$ROOT_PATH/yangping/checkpoints/modelevaluation/tnews/
OUTPUT_PATH=/$ROOT_PATH/yangping/nlp/modelevaluation/output/predict.json

DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.json \
        --valid_data dev.json \
        --test_data test.json \
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

DOCKER_PATH=/$ROOT_PATH/yangping/containers/pytorch21_06_py3_docker_image.sif
SCRIPT_PATH=/$ROOT_PATH/yangping/nlp/Fengshenbang-LM/fengshen/examples/finetune_classification.py

python3 $SCRIPT_PATH $options
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options


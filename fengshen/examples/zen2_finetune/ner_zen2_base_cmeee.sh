#!/bin/bash
#SBATCH --job-name=zen2_base_cmeee # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 
#SBATCH -o /cognitive_comp/ganruyi/experiments/ner_finetune/zen2_base_cmeee/%x-%j.log # output and error file name (%x=job name, %j=job id)


# export CUDA_VISIBLE_DEVICES='2'
export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions

MODEL_NAME=zen2_base

TASK=cmeee

ZERO_STAGE=1
STRATEGY=deepspeed_stage_${ZERO_STAGE}

ROOT_DIR=/cognitive_comp/ganruyi/experiments/ner_finetune/${MODEL_NAME}_${TASK}
if [ ! -d ${ROOT_DIR} ];then
  mkdir -p ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

DATA_DIR=/cognitive_comp/lujunyu/data_zh/NER_Aligned/CMeEE/
PRETRAINED_MODEL_PATH=/cognitive_comp/ganruyi/hf_models/zen/zh_zen_base_2.0

CHECKPOINT_PATH=${ROOT_DIR}/ckpt/
OUTPUT_PATH=${ROOT_DIR}/predict.json

DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.char.bio \
        --valid_data dev.char.bio \
        --test_data dev.char.bio \
        --train_batchsize 32 \
        --valid_batchsize 16 \
        --max_seq_length 256 \
        --task_name cmeee \
        "

MODEL_ARGS="\
        --learning_rate 3e-5 \
        --weight_decay 0.1 \
        --warmup_ratio 0.01 \
        --markup bio \
        --middle_prefix I- \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_f1 \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 100 \
        --save_weights_only True \
        --dirpath $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_f1:.4f} \
        "

TRAINER_ARGS="\
        --max_epochs 30 \
        --gpus 1 \
        --check_val_every_n_epoch 1 \
        --val_check_interval 100 \
        --default_root_dir $ROOT_DIR \
        "


options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --vocab_file $PRETRAINED_MODEL_PATH/vocab.txt \
        --do_lower_case \
        --output_save_path $OUTPUT_PATH \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
"
SCRIPT_PATH=//cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/models/zen2/fengshen_token_level_ft_task.py
/home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options

# SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif
# python3 $SCRIPT_PATH $options
# source activate base
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH /home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options
# /home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options


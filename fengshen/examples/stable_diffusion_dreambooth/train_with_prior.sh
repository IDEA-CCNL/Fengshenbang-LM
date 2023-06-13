#!/bin/bash
#SBATCH --job-name=taiyi-sd-dreambooth # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=2 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

# pwd=Fengshenbang-LM/fengshen/examples/pretrain_erlangshen
ROOT_DIR=../../workspace
# export CUDA_VISIBLE_DEVICES='2'
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

MODEL_NAME=taiyi-sd-dreambooth-prior
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=2
MICRO_BATCH_SIZE=2
# 如果你不用Deepspeed的话 下面的一段话都可以删掉 Begin
CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=1
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
     "fp16": {
        "enabled": true
    },
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
### End

INSTANCE_PROMPT="[小黄鸭]"
OUTPUT_DIR="saved_model_duck2"
INSTANCE_DIR="train_images_duck"

CLASS_PROMPT="小黄鸭"
CLASS_DIR="class_images_duck"

DATA_ARGS="\
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --instance_data_dir=$INSTANCE_DIR \
        --instance_prompt=$INSTANCE_PROMPT \
        --class_prompt=$CLASS_PROMPT \
        --class_data_dir=$CLASS_DIR \
        --with_prior_preservation --prior_loss_weight=1.0 \
        --num_class_images=200 \
        --resolution=512 \
        --sample_batch_size=1 \
        "

MODEL_ARGS="\
        --model_path $MODEL_ROOT_DIR/pretrain/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/ \
        --train_text_encoder \
        --learning_rate 1e-6 \
        --scheduler_type constant \
        "

MODEL_CHECKPOINT_ARGS="\
        --every_n_epochs 100 \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
        "

TRAINER_ARGS="\
        --max_epochs 200 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --precision 16 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 0 \
        "
# num_sanity_val_steps， limit_val_batches 通过这俩参数把validation关了

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "
# run local
# python train.py $options
# run on slurm
srun python train.py $options
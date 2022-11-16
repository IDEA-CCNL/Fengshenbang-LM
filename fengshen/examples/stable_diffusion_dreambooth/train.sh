#!/bin/bash
#SBATCH --job-name=taiyi-sd-dreambooth # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

# pwd=Fengshenbang-LM/fengshen/examples/pretrain_erlangshen
ROOT_DIR=../../workspace
# export CUDA_VISIBLE_DEVICES='7'
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

MODEL_NAME=taiyi-sd-dreambooth
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=1

MICRO_BATCH_SIZE=1
INSTANCE_PROMPT="小黄鸭"
OUTPUT_DIR="saved_model_tinyduck"
INSTANCE_DIR="train_images_duck"

DATA_ARGS="\
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --instance_data_dir=$INSTANCE_DIR \
        --instance_prompt=$INSTANCE_PROMPT \
        --resolution=512 \
        "

MODEL_ARGS="\
        --model_path $MODEL_ROOT_DIR/pretrain/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/ \
        --train_text_encoder \
        --learning_rate 1e-6 \
        --scheduler_type constant \
        --warmup_steps 100 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
        "

TRAINER_ARGS="\
        --max_steps 1200 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy ddp \
        --log_every_n_steps 100 \
        --precision 32 \
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
python train.py $options
# run on slurm
# srun python train.py $options
#!/bin/bash
  
#SBATCH -J generator_reasoning
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=32
#SBATCH --gres=gpu:8
#SBATCH -x dgx[045,050]
#SBATCH -o logs/generator_reasoning_math-gptj-%j.log
#SBATCH -e errs/generator_reasoning_math-gptj-%j.err
#SBATCH --requeue
#SBATCH --qos=preemptive

# set -x -e
export MASTER_PORT=$[RANDOM%10000+50000]
export TORCH_EXTENSIONS_DIR=/home/zhuxinyu/.cache/torch_extensions/py38_cu111
NUM_NODES=1
NUM_GPUS=8

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"
ROOT_DIR=/cognitive_comp/zhuxinyu/codes/reasoning_qa

TRAINER_ARGS="
    --gpus $NUM_GPUS \
    --save_dir $ROOT_DIR/outputs \
    --timestamp $TIMESTAMP \
    --precision 16 \
    --predict \
    --generator \
    --temp 0.7 \
    --num_sample 100 \
    --sample_len 400 \
    --strategy ddp \
"
    # --strategy deepspeed_stage_2 \

DATA_DIR=/cognitive_comp/zhuxinyu/datasets/grade-school-math/grade_school_math/data/
DATA_ARGS="
    --data_dir $DATA_DIR \
    --predict_data test.jsonl \
    --num_workers 32 \
    --predict_batch_size 20 \
    --task generator \
    --recreate_dataset \
"

MODEL_ARGS="
    --seed 20020206 \
    --model_type gpt \
    --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-05-17_13-13-39_loss_on_prefix_with_collate_fn/hf_pretrained_epoch9_step2340 \
"
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936 \
    # --model_name /cognitive_comp/zhuxinyu/pretrained_models/EleutherAI/gpt-j-6B \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-GSM-04-28_12-43/hf_pretrained_epoch0_step0 \
    # --prompt \

SCRIPTS_PATH=$ROOT_DIR/gpt_training_gsm8k.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
# srun  --nodes=$NUM_NODES --gres=gpu:$NUM_GPUS --cpus-per-gpu=32 bash -c 'python $CMD'
bash -c 'time python $CMD'
echo "END TIME: $(date)"

# SINGULARITY_PATH=/cognitive_comp/zhuxinyu/container/pytorch21_06_py3_docker_image_v2.sif
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python $CMD'


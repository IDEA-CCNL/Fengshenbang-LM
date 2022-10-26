#!/bin/bash
#SBATCH --job-name=pretrain_bart # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

# pwd=Fengshenbang-LM/fengshen/examples/pretrain_erlangshen
ROOT_DIR=../../workspace
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

MODEL_NAME=randeng-bart-base
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=1

MICRO_BATCH_SIZE=32

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
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
### End

DATA_ARGS="\
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        "
# 如果你有一批数据，可以参照IDEA-CCNL/PretrainCorpusDemo的格式处理，通过参数传入
# --train_file train.json
# --val_file val.json
# --test_file test.json

MODEL_ARGS="\
        --model_path $MODEL_ROOT_DIR/pretrain \
        --learning_rate 1e-4 \
        --weight_decay 1e-1 \
        --warmup_ratio 0.01 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
        "

TRAINER_ARGS="\
        --max_epoch 10 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 1 \
        --precision 16 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        "

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

python3 pretrain_bart.py $options
#srun -N $NNODES --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE --cpus-per-task=20 python3 pretrain_bart.py $options

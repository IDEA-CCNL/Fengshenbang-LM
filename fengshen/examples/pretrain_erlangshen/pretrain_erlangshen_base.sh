#!/bin/bash
#SBATCH --job-name=pretrain_bart # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050


ROOT_DIR=/cognitive_comp/gaoxinyu/experiment
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

NNODES=1
GPUS_PER_NODE=1

MODEL_NAME=erlangshen-base
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}

CONFIG_JSON="./config/${MODEL_NAME}.ds_config.json"

MICRO_BATCH_SIZE=8
ZERO_STAGE=1

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON

DATA_ARGS="\
        --num_workers 20 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --datasets_name IDEA-CCNL/PretrainCorpusDemo \
        "

MODEL_ARGS="\
        --model_path $MODEL_ROOT_DIR/pretrain \
        "

MODEL_CHECKPOINT_ARGS="\
        --every_n_train_steps 1 \
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

srun -N $NNODES --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE --cpus-per-task=20 python3 pretrain_erlangshen.py $options

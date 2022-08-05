#!/bin/bash
#SBATCH --job-name=pegasus-base_last # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)


set -x -e

echo "START TIME: $(date)"
MODEL_NAME=pegasus-base_test

config_json="./$MODEL_NAME.ds_config.json"
export MASTER_PORT=$[RANDOM%10000+40000]

MICRO_BATCH_SIZE=4

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
    "zero_optimization": {
        "stage": 1
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "params": {
            "betas": [
                0.9,
                0.999
            ],
            "eps": 1e-08,
            "lr": 1e-04,
            "weight_decay": 0.01
        },
        "type": "Adam"
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 1e-04,
            "warmup_min_lr": 1e-05,
            "total_num_steps": 80000000,
            "warmup_num_steps" : 50000
        },
        "type": "WarmupDecayLR"
    },
    "steps_per_print": 100,
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "zero_allow_untested_optimizer": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/dongxiaoqun/torch_extendsions

DATA_ARGS="\
        --datasets_name wudao_180g_512 \
        --num_workers 20 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --val_batchsize 8 \
        --test_batchsize 8  \
        --max_seq_length 512 \
        --val_datasets_field valid \
        "

MODEL_ARGS="\
        --model_path /cognitive_comp/dongxiaoqun/pretrained_model/pegasus-base/ \
        --learning_rate 1e-5 \
        --weight_decay 0.1 \
        --warmup_ratio 0.001 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 3 \
        --mode min \
        --every_n_train_steps 200 \
        --dirpath /cognitive_comp/dongxiaoqun/train_model/fengshen-$MODEL_NAME_debug/ckpt \
        --filename model-{step:02d}-{train_loss:.4f} \
        --save_last \
        "

TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 1 \
        --gpus 2 \
        --num_nodes 1 \
        --strategy ddp \
        --log_every_n_steps 100 \
        --val_check_interval 0.1 \
        --accumulate_grad_batches 8 \
        --default_root_dir /cognitive_comp/dongxiaoqun/train_model/fengshen-$MODEL_NAME_debug \
        --stopword_path /cognitive_comp/dongxiaoqun/pretrained_model/pegasus-large/stopwords \
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

SINGULARITY_PATH=/cognitive_comp/dongxiaoqun/software/docker/pytorch21_06_py3_docker_image_v2.sif
export SCRIPT_PATH=/cognitive_comp/dongxiaoqun/project/idea-ccnl/bug_fix/Fengshenbang-LM/fengshen/examples/pegasus/pretrain_pegasus.py

# python $SCRIPT_PATH $options
source activate
conda activate torchnew
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:2 --cpus-per-task=30 -o ${MODEL_NAME}-%J.log --jobid=226191 bash -c 'python3 $SCRIPT_PATH $options'

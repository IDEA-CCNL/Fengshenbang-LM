#!/bin/bash
#SBATCH --job-name=pipe_3.9B # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

INDEX_DATA_PREFIX=/cognitive_comp/gaoxinyu/data/WuDaoCorpus2.0_base_pretrain/wudao180G_bert_text_sentence
#TRAIN_SAMPLES=1843200000
TRAIN_SAMPLES=3706880
VAL_SAMPLES=3706880
TEST_SAMPLES=20480
MODEL_NAME=bert-3.9B

MICRO_BATCH_SIZE=2

config_json="./$MODEL_NAME.ds_config.json"
export MASTER_PORT=$[RANDOM%10000+30000]

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 32,
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
            "lr": 1e-06,
            "weight_decay": 0.01
        },
        "type": "Adam"
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 1e-06,
            "warmup_min_lr": 1e-06
        },
        "type": "WarmupLR"
    },
    "#flops_profiler": {
        "enabled": true,
        "profile_step": 100,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": true,
        "output_file": null
    },
    "steps_per_print": 100,
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "zero_allow_untested_optimizer": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/gaoxinyu/torch_extendsions
#export PL_FAULT_TOLERANT_TRAINING=1

DATA_ARGS="\
        --data_prefix $INDEX_DATA_PREFIX \
        --samples $TRAIN_SAMPLES $VAL_SAMPLES $TEST_SAMPLES \
        --num_workers 8 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --eval_batchsize 64 \
        --test_batchsize 8  \
        "

MODEL_ARGS="\
        --model_path /cognitive_comp/gaoxinyu/pretrained_model/$MODEL_NAME/ \
        --learning_rate 1e-6 \
        --weight_decay 0.1 \
        --warmup 0.001 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 3 \
        --mode min \
        --every_n_train_steps 8000 \
        --dirpath /cognitive_comp/gaoxinyu/ln_model/ckpt/fengshen-$MODEL_NAME \
        --filename model-{step:02d}-{train_loss:.4f} \
        "
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 1 \
        --gpus 1 \
        --num_nodes 1 \
        --max_steps 300 \
        --strategy deepspeed_stage_1 \
        --check_val_every_n_epoch 1 \
        --log_every_n_steps 25 \
        --accumulate_grad_batches 1 \
        --val_check_interval 1 \
        --resume
        --default_root_dir /cognitive_comp/gaoxinyu/ln_model/fengshen-$MODEL_NAME \
        "


options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
SCRIPT_PATH=/cognitive_comp/gaoxinyu/FS/fengshen/fengshen/examples/pretrain_erlangshen.py

srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH python3 $SCRIPT_PATH $options


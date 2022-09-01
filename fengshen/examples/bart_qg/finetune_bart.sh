#!/bin/bash
#SBATCH --job-name=randeng_pegasus_523M_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=30
#SBATCH -o %x-%j.log

set -x -e

MODEL_NAME=Randeng-BART-139M
RUN_NAME=bart_test
ROOT_DIR=/cognitive_comp/yangqi/logs/$RUN_NAME

config_json="$ROOT_DIR/$MODEL_NAME.ds_config.json"
export MASTER_PORT=$[RANDOM%10000+40000]

MICRO_BATCH_SIZE=8

cat <<EOT > $config_json
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "gradient_clipping": 1,
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
            "weight_decay": 1e-2
        },
        "type": "Adam"
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 1e-04,
            "warmup_min_lr": 1e-08,
            "total_num_steps": 100000,
            "warmup_num_steps" : 1000
        },
        "type": "WarmupDecayLR"
    },
    "steps_per_print": 1000,
    "zero_allow_untested_optimizer": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/yangqi/torch_extensions


DATA_ARGS=" \
        --datasets_name qag \
        --datasets_subname webqa \
        --sampler_type random \
        --tokenizer_type bart \
        --num_workers 8 \
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize 32  \
        --train_datasets_field train \
        --test_datasets_field test \
        --val_datasets_field validation \
        --max_seq_lengt 512 \
        --max_src_length 8 \
        --max_kno_length 440 \
        --max_tgt_length 64 \
        "

MODEL_ARGS="\
        --model_path /cognitive_comp/yangqi/model/$MODEL_NAME/ \
        --learning_rate 1e-5 \
        --weight_decay 0.1 \
        --warmup 0.0001 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 3 \
        --mode min \
        --save_last \
        --every_n_train_steps 1000 \
        --dirpath $ROOT_DIR/ckpt/ \
        --filename model-{step:02d}-{train_loss:.4f} \
        "
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 10 \
        --gpus 1 \
        --num_nodes 1 \
        --strategy deepspeed_stage_1 \
        --log_every_n_steps 100 \
        --val_check_interval 0.1 \
        --accumulate_grad_batches 1 \
        --default_root_dir $ROOT_DIR \
        "
#     --resume_from_checkpoint None\

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "
# test
export SCRIPT_PATH=/cognitive_comp/yangqi/project/Fengshenbang-LM/fengshen/examples/bart_qg/finetune_bart.py


# .02 debug mode
CUDA_LAUNCH_BLOCKING=3 CUDA_VISIBLE_DEVICES=7 python3 ${SCRIPT_PATH} $options > $ROOT_DIR/test.log

# slurm cluster mode
# SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python3 $SCRIPT_PATH $options'
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options > /cognitive_comp/yangqi/logs/$RUN_NAME/test.log


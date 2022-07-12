#!/bin/bash
#SBATCH --job-name=finetune_wenzhong # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx049

MODEL_NAME=Wenzhong-GPT2-110M

config_json="/cognitive_comp/yangqi/logs/wenzhong_dialo/$MODEL_NAME.ds_config.json"
export MASTER_PORT=$[RANDOM%10000+40000]

MICRO_BATCH_SIZE=8

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
            "total_num_steps": 536877,
            "warmup_num_steps" : 10000
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
export TORCH_EXTENSIONS_DIR=/cognitive_comp/yangqi/torch_extensions

TASK="query"
DATA_ARGS="\
        --task ${TASK} \
        --datasets_name dusinc_${TASK} \
        --pretrain_sp_tokenizer /cognitive_comp/yangqi/model/$MODEL_NAME/\
        --num_workers 30 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize 32  \
        --max_src_length 256 \
        --max_kno_length 128 \
        --max_tgt_length 128\
        --val_datasets_field test \
        "
# --masked_lm_prob 0.15 \
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
        --every_n_train_steps 10000 \
        --dirpath /cognitive_comp/yangqi/logs/wenzhong_dialo/$MODEL_NAME/ckpt_${TASK}/ \
        --filename model-{step:02d}-{train_loss:.4f} \
        "
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 20 \
        --gpus 1 \
        --num_nodes 1 \
        --strategy deepspeed_stage_1 \
        --log_every_n_steps 100 \
        --val_check_interval 0.1 \
        --accumulate_grad_batches 1 \
        --resume_from_checkpoint None\
        --default_root_dir /cognitive_comp/yangqi/logs/wenzhong_dialo/ \
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "
# test
export SCRIPT_PATH=/home/yangqi/code/Fengshenbang-LM/fengshen/examples/wenzhong_dialo/finetune_wenzhong_dialogue.py


# .02 debug mode
CUDA_LAUNCH_BLOCKING=3 python3 $SCRIPT_PATH $options > /cognitive_comp/yangqi/logs/wenzhong_dialo/test.log

# slurm cluster mode
#SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
#singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python3 $SCRIPT_PATH $options'
#singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options > /cognitive_comp/yangqi/logs/wenzhong_dialo/test.log

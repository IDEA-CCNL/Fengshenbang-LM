#!/bin/bash
#SBATCH --job-name=pretrain-wav2vec2 # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=4 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

MODEL_NAME=wav2vec2-base
config_json="./$MODEL_NAME.ds_config.json"

export MASTER_PORT=20992
MICRO_BATCH_SIZE=8
ZERO_STAGE=1
HOME_PATH=/cognitive_comp/zhuojianheng/experiment
MODEL_PATH=/cognitive_comp/zhuojianheng/pretrained_model/patrickvonplaten/wav2vec2-base-v2
DATA_DIR=/cognitive_comp/common_data/librispeech_tsv/datas

export TORCH_EXTENSIONS_DIR=/cognitive_comp/zhuojianheng/torch_extendsions


# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
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
    "tensorboard": {
        "enabled": true,
        "output_path": "${HOME_PATH}/fengshen-${MODEL_NAME}/ds-tb-logs",
        "job_name": "${MODEL_NAME}"
    },
    "#flops_profiler": {
        "enabled": true,
        "profile_step": 200,
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

DATA_ARGS="\
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --val_batchsize 16 \
        --test_batchsize 16  \
        --val_datasets_field valid \
        --test_datasets_field valid \
        --sampler_type random \
        --data ${DATA_DIR} \
        --max_sample_size 250000 \
        --min_sample_size 32000 \
        --normalize False \
        --enable_padding True \
        "

MODEL_ARGS="\
        --model_path ${MODEL_PATH} \
        --learning_rate 1e-4 \
        --weight_decay 1e-2 \
        --warmup_ratio 0.01 \
        --adam_beta1 0.9 \
        --adam_beta2 0.99 \
        --adam_epsilon 1e-6 \
        "

        # --pred_masked_weight 1.0 \
        # --loss_weights 10 \
MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 0 \
        --mode min \
        --every_n_train_steps 10000 \
        --dirpath ${HOME_PATH}/fengshen-${MODEL_NAME}/ckpt/\
        --filename model-{step:02d}-{train_loss:.4f} \
        --every_n_epochs 0 \
        --save_last \
        "

# deepspeed_stage_${ZERO_STAGE} \
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 10 \
        --gpus 4 \
        --num_nodes 1 \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --val_check_interval 50 \
	    --limit_val_batches 10 \
        --accumulate_grad_batches 8 \
        --precision 16 \
        --ckpt_path ${HOME_PATH}/fengshen-${MODEL_NAME}/ckpt/last.ckpt \
        --default_root_dir ${HOME_PATH}/fengshen-${MODEL_NAME}/ \
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

export SCRIPT_PATH=pretrain_wav2vec2.py
srun python3 $SCRIPT_PATH $options
# eval python3 -m debugpy --listen localhost:5876 --wait-for-client $SCRIPT_PATH $options

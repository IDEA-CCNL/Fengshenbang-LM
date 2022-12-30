#!/bin/bash
#SBATCH --job-name=pretrain_bart # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

MODEL_NAME=hubert-base-ls960
config_json="./$MODEL_NAME.ds_config.json"
export MASTER_PORT=29503
MICRO_BATCH_SIZE=8
ZERO_STAGE=1

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
        "output_path": "/data/training_model/fengshen-${MODEL_NAME}/ds-tb-logs",
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
export TORCH_EXTENSIONS_DIR=/home/gaoxinyu/torch_extendsions

DATA_DIR=/data/common_data/librispeech_tsv/datas
LABELS_DIR=/data/common_data/librispeech_tsv/labels

DATA_ARGS="\
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --val_batchsize 32 \
        --test_batchsize 8  \
        --val_datasets_field valid \
        --test_datasets_field valid \
        --sampler_type random \
        --data ${DATA_DIR} \
        --label_dir ${LABELS_DIR} \
        --labels km \
        --label_rate 100 \
        --max_sample_size 250000 \
        --min_sample_size 32000 \
        --pad_audio False \
        --random_crop True \
        --normalize False \
        "

MODEL_ARGS="\
        --model_path /data/pretrained_model/$MODEL_NAME/ \
        --learning_rate 1e-4 \
        --weight_decay 1e-2 \
        --warmup_ratio 0.01 \
        --pred_masked_weight 1.0 \
        --loss_weights 10 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 0 \
        --mode min \
        --every_n_train_steps 10000 \
        --dirpath /data/training_model/ckpt/fengshen-$MODEL_NAME \
        --filename model-{step:02d}-{train_loss:.4f} \
        --every_n_epochs 0 \
        --save_last \
        --not_save_on_train_epoch_end \
        "

# deepspeed_stage_${ZERO_STAGE} \
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 10 \
        --gpus 2 \
        --num_nodes 1 \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --val_check_interval 500 \
	    --limit_val_batches 10 \
        --accumulate_grad_batches 1 \
        --precision 16 \
        --ckpt_path /data/training_model/ckpt/fengshen-${MODEL_NAME}/last.ckpt \
        --default_root_dir /data/training_model/fengshen-$MODEL_NAME \
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

export SCRIPT_PATH=pretrain_hubert.py

eval python3 -m debugpy --listen localhost:53005 --wait-for-client $SCRIPT_PATH $options

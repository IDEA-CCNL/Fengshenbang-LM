#!/bin/bash
#SBATCH --job-name=wav2vec2-ctc # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=2 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2 # number of gpus per node
#SBATCH -o ./output/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

CHECK_POINT_NAME=hf_pretrained_epoch47_step400000
MODEL_NAME=wav2vec2-base-wenet-ctc-${CHECK_POINT_NAME}
config_json="./output/$MODEL_NAME.ds_config.json"

export MASTER_PORT=20989
MICRO_BATCH_SIZE=8
ZERO_STAGE=1
HOME_PATH=/cognitive_comp/zhuojianheng/experiment
MODEL_PATH=/cognitive_comp/zhuojianheng/pretrained_model/wav2vec2-base-ctc-wenet
DATA_DIR=/cognitive_comp/zhuojianheng/data/wenet/S
PRETRAINED_PATH=/cognitive_comp/zhuojianheng/experiment/fengshen-wav2vec2-base-wenet/ckpt/${CHECK_POINT_NAME}

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
        --learning_rate 3e-5 \
        --weight_decay 1e-2 \
        --warmup_ratio 0.1 \
        --scheduler_type constant_with_warmup \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-8 \
        --eval_metrics wer cer \
        --sampler_type fairseq \
        --required_batch_size_multiple 1 \
        --max_tokens 3200000
        --pretrained_model $PRETRAINED_PATH
        --architecture wav2vec2
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
        --save_on_train_epoch_end \
        "

# deepspeed_stage_${ZERO_STAGE} \
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --gpus 2 \
        --num_nodes 1 \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --val_check_interval 1000 \
	    --limit_val_batches 10 \
        --accumulate_grad_batches 4 \
        --precision 16 \
        --ckpt_path ${HOME_PATH}/fengshen-${MODEL_NAME}/ckpt/last.ckpt \
        --default_root_dir ${HOME_PATH}/fengshen-${MODEL_NAME}/ \
        --replace_sampler_ddp false \
        --max_steps 80000
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

CODE_HOME=/cognitive_comp/zhuojianheng/work/git/Fengshenbang-LM/fengshen/examples/wav2vec2
export SCRIPT_PATH=${CODE_HOME}/ctc_finetune.py
# python3 $SCRIPT_PATH $options
srun python3 $SCRIPT_PATH $options
# eval python3 -m debugpy --listen localhost:5876 --wait-for-client $SCRIPT_PATH $options

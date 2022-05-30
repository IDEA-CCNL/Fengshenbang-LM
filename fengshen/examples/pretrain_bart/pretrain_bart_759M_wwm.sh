#!/bin/bash
#SBATCH --job-name=pretrain_bart # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

set u+x

MODEL_NAME=bart-759M-wwm
INDEX_DATA_PREFIX=/cognitive_comp/gaoxinyu/data/WuDaoCorpus2.0_base_pretrain/wudao180G_bert_text_sentence
TRAIN_SAMPLES=1843200000
#TRAIN_SAMPLES=3706880
VAL_SAMPLES=3706880
TEST_SAMPLES=20480
config_json="./$MODEL_NAME.ds_config.json"
export MASTER_PORT=29501
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
    "steps_per_print": 100,
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "zero_allow_untested_optimizer": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/gaoxinyu/torch_extendsions

DATA_ARGS="\
        --data_prefix $INDEX_DATA_PREFIX \
        --samples $TRAIN_SAMPLES $VAL_SAMPLES $TEST_SAMPLES \
        --num_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --eval_batchsize 32 \
        --test_batchsize 8  \
        "

MODEL_ARGS="\
        --model_path /cognitive_comp/gaoxinyu/pretrained_model/$MODEL_NAME/ \
        --learning_rate 5e-6 \
        --weight_decay 1e-1 \
        --warmup_ratio 0.001 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 5 \
        --mode min \
        --every_n_train_steps 3200 \
        --dirpath /cognitive_comp/gaoxinyu/ln_model/ckpt/fengshen-$MODEL_NAME \
        --filename model-{step:02d}-{train_loss:.4f} \
        --every_n_epochs 0 \
        --save_last \
        "
TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_steps 7000000 \
        --gpus 8 \
        --num_nodes 4 \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 1 \
        --val_check_interval 0.01 \
        --accumulate_grad_batches 8 \
        --limit_val_batches 100 \
        --precision 16 \
        --ckpt_path /cognitive_comp/gaoxinyu/ln_model/ckpt/fengshen-${MODEL_NAME}/last.ckpt \
        --default_root_dir /cognitive_comp/gaoxinyu/ln_model/fengshen-$MODEL_NAME \
        --replace_sampler_ddp False \
        "


export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
export SCRIPT_PATH=/cognitive_comp/gaoxinyu/github/Fengshenbang-LM/fengshen/examples/pretrain_bart/pretrain_bart_wwm.py

#srun -N 4 --ntasks-per-node=8 --cpus-per-task=20 --gres=gpu:8 singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'echo $SLURMD_NODENAME:$(($MASTER_PORT+$SLURM_PROCID+1));python3 -m debugpy --listen $SLURMD_NODENAME:$(($MASTER_PORT+$SLURM_PROCID+1)) $SCRIPT_PATH $options'
srun -N 4 --ntasks-per-node=8 --cpus-per-task=30 --gres=gpu:8 singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python3 $SCRIPT_PATH $options'
#srun -N 4 --ntasks-per-node=8 --cpus-per-task=30 --gres=gpu:8 bash -c 'echo $(fuser -v /dev/nvidia${SLURM_LOCALID} 2>/dev/null)'
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options


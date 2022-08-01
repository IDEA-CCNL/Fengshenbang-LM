#!/bin/bash
  
#SBATCH -J verifier_math-gpt-j
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=32
#SBATCH --gres=gpu:8
#SBATCH -x dgx[040,045,050,051,073]
#SBATCH -o logs/verifier_math-gptj-%j.log
#SBATCH -e errs/verifier_math-gptj-%j.err

# set -x -e
export MASTER_PORT=$[RANDOM%10000+50000]
export TORCH_EXTENSIONS_DIR=/home/zhuxinyu/.cache/torch_extensions/py38_cu111
NUM_NODES=1
NUM_GPUS=8

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"
ROOT_DIR=/cognitive_comp/zhuxinyu/codes/reasoning_qa

ZERO_STAGE=2

config_json="$ROOT_DIR/ds_config.$SLURM_JOBID.json"
# config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_micro_batch_size()
        # "offload_param":{
        #   "device":"cpu",
        #   "pin_memory":true
        # },
    # "optimizer":{
    #     "type":"Adam",
    #     "params":{
    #         "betas": [
    #             0.9,
    #             0.95
    #         ],
    #         "eps": 1e-8,
    #         "lr": 1e-6,
    #         "weight_decay":0
    #     }
    # },
    # "scheduler":{
    #     "type":"WarmupDecayLR",
    #     "params":{
    #         "warmup_min_lr":1e-6,
    #         "warmup_max_lr":1e-5,
    #         "warmup_num_steps": 500,
    #         "total_num_steps": 4574
    #     }
    # },

    # "gradient_clipping":1,
cat <<EOT > $config_json
{
    "train_micro_batch_size_per_gpu":2,
    "steps_per_print":10,
    "zero_optimization":{
        "stage": $ZERO_STAGE,
        "offload_optimizer":{
          "device":"cpu",
          "pin_memory":true
        },
        "overlap_comm":true,
        "contiguous_gradients":true,
        "sub_group_size":1000000000,
        "stage3_max_live_parameters":1000000000,
        "stage3_max_reuse_distance":1000000000,
        "stage3_gather_fp16_weights_on_model_save":true
    },
    "zero_allow_untested_optimizer":false,
    "fp16":{
        "enabled":true,
        "loss_scale":0,
        "loss_scale_window":1000,
        "initial_scale_power": 16,
        "hysteresis":2,
        "min_loss_scale":1
    },
    "activation_checkpointing":{
        "partition_activations":false,
        "contiguous_memory_optimization":false
    },
    "wall_clock_breakdown":false
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$config_json

TRAINER_ARGS="
    --max_epochs 2 \
    --gpus $NUM_GPUS \
    --log_every_n_steps 1 \
    --precision 16 \
    --save_dir $ROOT_DIR/verifier_outputs \
    --save_top_k 5 \
    --monitor avg_train_loss \
    --mode min \
    --timestamp $TIMESTAMP \
    --gradient_clip_val 1.0 \
    --train \
    --num_nodes $NUM_NODES \
    --accumulate_grad_batches 1 \
    --strategy deepspeed_stage_2 \
"
    # --strategy ddp \
    # --predict \
    # --val_check_interval 0.5 \
    # --patience 3 \
    # --check_val_every_n_epoch 1 \

DATA_DIR=/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936/07-11-mcts_verifier_file_capacity_1000_trainset
DATA_ARGS="
    --data_dir $DATA_DIR \
    --num_workers 32 \
    --train_data merged_07-11_12_13-mcts_verifier_file.jsonl_verifier_acc_7428_voting_acc_7330_solve_rate_8541 \
    --micro_batch_size 2 \
    --task verifier \
    --recreate_dataset \
"
    # --valid_batch_size 4 \
    # --train_data tiny_train.jsonl \
    # --source_max_token_len 512 \
    # --valid_data 05-11_00-03-generator_solution_merged_seed_20020206.jsonlprocessed \

MODEL_ARGS="
    --seed 19990303 \
    --model_type gpt \
    --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/init_from_gpt-j-6B-GSM-05-10_12-33_hf_pretrained_epoch1_74w_data_for_train/hf_pretrained_epoch0_step46706 \
    --lr 1e-6 \
    --l2 0. \
    --warmup 0.1 \
    --show_training_ex 100 \
    --scheduler linear \
"
    # --lm_objective \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936 \
    # --model_name /cognitive_comp/zhuxinyu/pretrained_models/gpt2 \
    # --model_name /cognitive_comp/zhuxinyu/pretrained_models/EleutherAI/gpt-j-6B \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-04-18_15-45/hf_pretrained_epoch0_step0 \
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-04-20_00-28/last-epoch=14-avg_val_loss=1.3233.ckpt \
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-04-17_18-26-mapper_openai/last-epoch=08-avg_val_loss=1.0418.ckpt \
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-04-18_15-45/last-epoch=08-avg_val_loss=1.1399.ckpt \


SCRIPTS_PATH=$ROOT_DIR/verifier_training_gsm8k.py
# SCRIPTS_PATH=$ROOT_DIR/verifier_data_model.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
# --jobid $SLURM_JOBID
# --ntasks-per-node=8
# --cpus-per-gpu=32
# srun  --nodes=$NUM_NODES --gres=gpu:$NUM_GPUS --cpus-per-gpu=32 bash -c 'python $CMD'
bash -c 'python $CMD'
echo "END TIME: $(date)"

# SINGULARITY_PATH=/cognitive_comp/zhuxinyu/container/pytorch21_06_py3_docker_image_v2.sif
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python $CMD'


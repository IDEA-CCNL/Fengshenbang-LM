#!/bin/bash
  
#SBATCH -J test_verifier_math
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=32
#SBATCH --gres=gpu:2
#SBATCH -o logs/test_verifier_math-gptj-%j.log
#SBATCH -e errs/test_verifier_math-gptj-%j.err
#SBATCH --requeue
#SBATCH --qos=preemptive

# set -x -e
export MASTER_PORT=$[RANDOM%10000+50000]
export TORCH_EXTENSIONS_DIR=/home/zhuxinyu/.cache/torch_extensions/py38_cu111
NUM_NODES=1
NUM_GPUS=2

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
cat <<EOT > $config_json
{
    "train_micro_batch_size_per_gpu":1,
    "steps_per_print":10,
    "gradient_clipping":1,
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
    "optimizer":{
        "type":"Adam",
        "params":{
            "betas": [
                0.9,
                0.95
            ],
            "eps": 1e-8,
            "lr": 1e-6,
            "weight_decay":0
        }
    },
    "scheduler":{
        "type":"WarmupDecayLR",
        "params":{
            "warmup_min_lr":1e-6,
            "warmup_max_lr":1e-5,
            "warmup_num_steps": 4454,
            "total_num_steps": 44549
        }
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
    --gpus $NUM_GPUS \
    --precision 16 \
    --save_dir $ROOT_DIR/verifier_outputs \
    --timestamp $TIMESTAMP \
    --predict \
    --num_nodes $NUM_NODES \
    --strategy ddp \
"
    # --strategy deepspeed_stage_2 \
    # --predict \

DATA_DIR=/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936/07-03-mcts_verifier_file_capacity_1000
DATA_ARGS="
    --data_dir $DATA_DIR \
    --num_workers 32 \
    --predict_data merged_07-03-20-5x-mcts_verifier_file.jsonl_verifier_acc_5716_voting_acc_5550_solve_rate_7839 \
    --predict_batch_size 64 \
    --task verifier_score \
    --recreate_dataset \
"

MODEL_ARGS="
    --seed 19990303 \
    --model_type deberta \
    --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/hf_pretrained_epoch2_step17517-Verifier-GSM-07-30_15-15-29/hf_pretrained_epoch2_step8766 \
"
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/init_from_gpt-j-6B-GSM-05-10_12-33_hf_pretrained_epoch1/epoch=00-avg_train_loss=0.0136.ckpt/pytorch_model.bin \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/init_from_gpt2-large-GSM-05-13_12-57-03_hf_pretrained_epoch13/hf_pretrained_epoch0_step22275 \
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/init_from_gpt2-large-GSM-05-13_12-57-03_hf_pretrained_epoch13/epoch=00-avg_train_loss=0.1922.ckpt \
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


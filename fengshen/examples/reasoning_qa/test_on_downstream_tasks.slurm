#!/bin/bash
  
#SBATCH -J predict_reasoning
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=32
#SBATCH --gres=gpu:1
#SBATCH -x dgx[045,050,051,073]
#SBATCH -o logs/predict_reasoning_math-gptj-%j.log
#SBATCH -e errs/predict_reasoning_math-gptj-%j.err
#SBATCH --requeue
#SBATCH --qos=preemptive

# set -x -e
export MASTER_PORT=$[RANDOM%10000+50000]
export TORCH_EXTENSIONS_DIR=/home/zhuxinyu/.cache/torch_extensions/py38_cu111
NUM_NODES=1
NUM_GPUS=1

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"
ROOT_DIR=/cognitive_comp/zhuxinyu/codes/reasoning_qa

ZERO_STAGE=2

config_json="$ROOT_DIR/ds_config.$SLURM_JOBID.json"
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
    --save_dir $ROOT_DIR/outputs \
    --timestamp $TIMESTAMP \
    --precision 16 \
    --predict \
    --sample_len 200 \
    --strategy deepspeed_stage_2 \
"
    # --strategy ddp \

# DATA_DIR=/cognitive_comp/zhuxinyu/datasets/grade-school-math/grade_school_math/data/
# DATA_DIR=/cognitive_comp/zhuxinyu/datasets/MWPToolkit/dataset/mawps-single/
# DATA_DIR=/cognitive_comp/zhuxinyu/datasets/MWPToolkit/dataset/asdiv-a/
# DATA_DIR=/cognitive_comp/zhuxinyu/datasets/MultiArith/
DATA_DIR=/cognitive_comp/zhuxinyu/datasets/Reasoning_dataset/
DATA_ARGS="
    --data_dir $DATA_DIR \
    --num_workers 32 \
    --predict_batch_size 32 \
    --predict_data "SingleEq-gsm8k-format.jsonl" \
    --task predictor \
    --recreate_dataset \
"
    # --predict_data "AddSub-gsm8k-format.jsonl" \
    # --predict_data "SingleOp-gsm8k-format.jsonl" \
    # --predict_data "multi_arith-gsm8k-format.jsonl" \
    # --predict_data "merged_asdiv-a-gsm8k-format.jsonl" \
    # --predict_data "svamp-gsm8k-format.jsonl" \
    # --predict_data "merged-mawps-s.jsonl" \
    # --predict_data test.jsonl \

MODEL_ARGS="
    --seed 19990303 \
    --model_type gpt \
    --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch0_step468 \
"
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-05-17_13-13-39_loss_on_prefix_with_collate_fn/hf_pretrained_epoch9_step2340 \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-04-18_15-45/hf_pretrained_epoch0_step0 \
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-04-18_15-45/last-epoch=08-avg_val_loss=1.1399.ckpt \
    # --model_name /cognitive_comp/zhuxinyu/pretrained_models/EleutherAI/gpt-j-6B \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-GSM-04-28_12-43/hf_pretrained_epoch0_step0 \
    # --continue_train_from_ckpt /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-GSM-04-28_12-43/last-epoch=19-avg_val_loss=0.9739.ckpt \
    # --prompt \

SCRIPTS_PATH=$ROOT_DIR/gpt_training_gsm8k.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
# srun  --nodes=$NUM_NODES --gres=gpu:$NUM_GPUS --cpus-per-gpu=32 bash -c 'python $CMD'
bash -c 'time python $CMD'
echo "END TIME: $(date)"

# SINGULARITY_PATH=/cognitive_comp/zhuxinyu/container/pytorch21_06_py3_docker_image_v2.sif
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c 'python $CMD'


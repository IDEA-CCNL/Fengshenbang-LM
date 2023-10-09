#!/usr/bin/env bash
#SBATCH --job-name=chimed_sft_hf # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # total number of tasks across all nodes
#SBATCH --cpus-per-task=4 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:8 # number of gpus per node
#SBATCH -p pol # -preempted

#SBATCH -o %x-%j.log # output and error log file names (%x for job id)

ROOT_DIR=../../workspace
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

MODEL_NAME=ziya_llama2_13b_medical
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir -p ${MODEL_ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=8
MICRO_BATCH_SIZE=2

# 如果你不用Deepspeed的话 下面的一段话都可以删掉 Begin
CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=2

cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "activation_checkpointing": {
      "partition_activations": true,
      "contiguous_memory_optimization": true,
      "number_checkpoints": 20
    },
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
### End

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -n 1 -i 40000-65535)
# MASTER_PORT=51200
echo $MASTER_ADDR
echo $MASTER_PORT


export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    "
CODE_PATH="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/examples/ziya_llama/finetune_ziya_with_hf.py"
# DATA_PATH="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_cmd_sft.json"
DATA_PATH="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_cmd10w_mc_chimed.json"

VALID_DATA_PATH="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/valid_cmd_sft.json"
MODEL_PATH="/cognitive_comp/common_checkpoint/llama2_hf_13b_step191000"
OUTPUT_PATH="$MODEL_ROOT_DIR/output_v2"
# OLD_PATH="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/output/checkpoint-23000/"

export WANDB_PROJECT=$MODEL_NAME
export WANDB_LOG_MODEL='all'
export run_name='medical_sft_hf_v2'

# --model_name_or_path $MODEL_PATH \


srun --jobid $SLURM_JOBID bash -c `python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT $CODE_PATH \
--data_path $DATA_PATH \
--eval_data_path $VALID_DATA_PATH \
--output_dir $OUTPUT_PATH \
--model_name_or_path $MODEL_PATH \
--model_max_length 1024 \
--num_train_epochs 2 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--lr_scheduler_type cosine \
--adam_beta1 0.9 \
--adam_beta2 0.98 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--weight_decay 1e-4 \
--warmup_ratio 0.01 \
--logging_steps 1 \
--log_level "debug" \
--bf16 True \
--deepspeed $CONFIG_JSON \
--do_train \
--do_eval \
--evaluation_strategy "steps" \
--save_steps 5000 \
--eval_steps 1000 \
--report_to "wandb" \
--run_name medical_sft_hf \
--gradient_checkpointing True \
`


# 
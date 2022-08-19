#!/bin/bash
#SBATCH --job-name=randeng_t5_77M_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2               # number of gpus
#SBATCH --cpus-per-task=30
#SBATCH -o %x-%j.log

set -x -e

echo "START TIME: $(date)"
MODEL_NAME=randeng_t5_77M_summary_test2
MICRO_BATCH_SIZE=64
ROOT_DIR=/cognitive_comp/dongxiaoqun/finetune/${MODEL_NAME}
if [ ! -d ${ROOT_DIR} ];then
  mkdir ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

output_save_path=$ROOT_DIR/${MODEL_NAME}.json
if [ -f ${output_save_path} ];then
  echo ${output_save_path} exist, rm it!!!!!!!!!!!!!!!!!
  rm ${output_save_path}
fi
ZERO_STAGE=1

config_json="${ROOT_DIR}/ds_config.${MODEL_NAME}.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 100,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "contiguous_gradients": false,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 500000000
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-4,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "params": {
      "warmup_max_lr": 1e-04,
      "warmup_min_lr": 1e-05,
      "total_num_steps": 60000,
      "warmup_num_steps" : 500
    },
    "type": "WarmupDecayLR"  
  },
  "zero_allow_untested_optimizer": false,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/dongxiaoqun/torch_extendsions
# export MASTER_PORT=$[RANDOM%10000+30000]
# export PL_FAULT_TOLERANT_TRAINING=1

TRAINER_ARGS="
    --max_epochs 2 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy deepspeed_stage_${ZERO_STAGE} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --monitor val_loss \
    --mode min \
    --save_last \
    --every_n_train_steps 0 \
    --val_check_interval 0.1 \
"

prompt="summary:"
DATA_ARGS="
    --datasets_name lcsts \
    --num_workers 30 \
    --train_batchsize $MICRO_BATCH_SIZE \
    --val_batchsize $MICRO_BATCH_SIZE \
    --test_batchsize $MICRO_BATCH_SIZE \
    --max_enc_length 128 \
    --max_dec_length 64 \
    --val_datasets_field val \
    --prompt $prompt \
"
# --prompt $prompt \
MODEL_ARGS="
    --pretrained_model_path /cognitive_comp/ganruyi/experiments/randeng_t5_77M/ckpt/hf_pretrained_epoch0_step183100 \
    --output_save_path $ROOT_DIR/randeng_t5_77M_predict_lcsts.json \
"

SCRIPTS_PATH=/cognitive_comp/dongxiaoqun/debug/Fengshenbang-LM/fengshen/examples/summary/seq2seq_summary.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "
echo $CMD
# python $CMD

source activate
conda activate torchnew
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=30 -o ${MODEL_NAME}-%J.log --jobid=229623 bash -c 'python3 $SCRIPT_PATH $CMD'

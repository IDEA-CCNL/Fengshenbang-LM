#!/bin/bash
#SBATCH --job-name=predict-cmrc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --cpus-per-task=4 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o /cognitive_comp/hejunqing/projects/CMRC/models/%x-%j.log
#SBATCH -e /cognitive_comp/hejunqing/projects/CMRC/models/%x-%j.err

set -x -e

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=8

ROOT_DIR=/cognitive_comp/hejunqing/projects/CMRC/models/v1_bs8/
DOWNLOAD_MODEL_PATH=/cognitive_comp/hejunqing/projects/CMRC/huggingface/

if [ ! -d ${ROOT_DIR} ];then
  mkdir ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

ZERO_STAGE=1

config_json="$ROOT_DIR/ds_config.randeng_t5_dialog_784M.$SLURM_JOBID.json"
export MASTER_PORT=$[RANDOM%10000+30000]

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
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "total_num_steps": 1500,
      "warmup_num_steps" : 150
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
export TORCH_EXTENSIONS_DIR=/cognitive_comp/hejunqing/tmp/torch_extendsions
# strategy=ddp
strategy=deepspeed_stage_1

TRAINER_ARGS="
    --max_epochs 10 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy ${strategy} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 5 \
    --every_n_train_steps 100\
    --monitor val_rougeL_fmeasure \
    --mode max \
    --save_last \
    --check_val_every_n_epoch 1 \
    --dataset_num_workers 4 \
    --dataloader_num_workers 4 \
    --replace_sampler_ddp False \
    --accumulate_grad_batches 2 \
    --formator t5style \
    --filename model-{epoch:02d}-{val_loss:.4f}-{val_rougeL_fmeasure:.3f} \
    --do_eval_only \
    --prediction_res_path $ROOT_DIR/predictions_sampling.txt \
    --decode_strategy sampling
"

DATA_DIR=cmrc

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --train_data_path ${DATA_DIR} \
    --max_seq_length 512 \
    --max_knowledge_length 425 \
    --max_target_length 128
"
MODEL_ARGS="
    --pretrained_model_path $DOWNLOAD_MODEL_DIR\
    --tokenizer_type t5_tokenizer \
"

SCRIPTS_PATH=/cognitive_comp/hejunqing/projects/Fengshenbang-LM/fengshen/examples/qa_t5/finetune_t5_cmrc.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
# conda activate fs
# export CUDA_VISIBLE_DEVICES=5
srun python $CMD

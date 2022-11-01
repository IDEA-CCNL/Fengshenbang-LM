#!/bin/bash
#SBATCH --job-name=predict-cmrc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --cpus-per-task=4 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o $YOUR_SLURM_LOG_PATH/%x-%j.log
#SBATCH -e $YOUR_SLURM_LOG_PATH/%x-%j.err

#
set -x -e

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=8

ROOT_DIR=$YOUR_PROJECT_DIR
DOWNLOAD_MODEL_PATH=$YOUR_PROJECT_DIR/Randeng-T5-784M-QA-Chinese/
#YOUR_MODEL_DIR

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
  "zero_allow_untested_optimizer": false,
  "wall_clock_breakdown": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=$YOUR_HOME/tmp/torch_extendsions
# strategy=ddp
strategy=deepspeed_stage_1

TRAINER_ARGS="
    --max_epochs 10 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy ${strategy} \
    --default_root_dir $ROOT_DIR \
    --save_ckpt_path $ROOT_DIR/ckpt \
    --save_top_k 5 \
    --every_n_train_steps 100\
    --monitor val_rougeL_fmeasure \
    --mode max \
    --save_last \
    --check_val_every_n_epoch 1 \
    --num_workers 4 \
    --dataloader_workers 4 \
    --replace_sampler_ddp False \
    --accumulate_grad_batches 2 \
    --formator t5style \
    --filename model-{epoch:02d}-{val_loss:.4f}-{val_rougeL_fmeasure:.3f} \
    --do_eval_only \
    --prediction_res_path $ROOT_DIR/predictions_sampling.txt \
    --decode_strategy sampling \
    --precision 16 \
"

TEST_FILE_PATH=$YOUR_DATA_FILE

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --val_batchsize $MICRO_BATCH_SIZE \
    --test_file $TEST_FILE_PATH \
    --max_seq_length 512 \
    --max_knowledge_length 425 \
    --max_target_length 128
"
MODEL_ARGS="
    --pretrained_model_path $DOWNLOAD_MODEL_PATH\
    --tokenizer_type t5_tokenizer \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --warmup_ratio 0.1 \
    --sheduler_type polynomial \
    --min_learning_rate 1e-5 \
"

SCRIPTS_PATH=$YOUR_PROJECT_DIR/Fengshenbang-LM/fengshen/examples/qa_t5/finetune_t5_cmrc.py

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

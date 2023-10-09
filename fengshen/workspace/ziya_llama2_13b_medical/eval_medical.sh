#!/usr/bin/env bash
#SBATCH --job-name=eval_medical # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=4 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pol-preempted # -preempted

#SBATCH -o %x-%j.log # output and error log file names (%x for job id)

ROOT_DIR=../../workspace
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

MODEL_NAME=ziya_llama2_13b_medical
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir -p ${MODEL_ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=1
MICRO_BATCH_SIZE=2

# 如果你不用Deepspeed的话 下面的一段话都可以删掉 Begin
CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=2
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
        },
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

DATA_ARGS="\
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --train_file /cognitive_comp/ganruyi/datasets/medical_data/cMedQA2/test_processed.json.test \
        --val_file /cognitive_comp/ganruyi/datasets/medical_data/cMedQA2/test_processed.json.test \
        --test_file /cognitive_comp/ganruyi/datasets/medical_data/cMedQA2/test_processed.json.test \
        --use_mpu \
        "
MODEL_PATH="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/baseline_ckpt/hf_pretrained_epoch1_step3138"
MODEL_ARGS="\
        --model_path $MODEL_PATH \
        --tokenizer_path $MODEL_PATH \
        --learning_rate 5e-5 \
        --min_learning_rate 1e-5 \
        --weight_decay 0.1 \
        --warmup_ratio 0.1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --max_seq_length 1024 \
        --model_parallel_size 1 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --every_n_train_steps 100  \
        --save_ckpt_path ${MODEL_ROOT_DIR}/baseline_ckpt \
        "
#         --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \

prediction_res_path="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/result_cmedqa_v2.out"
TRAINER_ARGS="\
        --max_epoch 1 \
        --accelerator gpu \
        --devices $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --log_every_n_steps 1 \
        --precision 16 \
        --accumulate_grad_batches 1 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --check_val_every_n_epoch 1 \
        --do_eval_only \
        --prediction_res_path ${prediction_res_path} \
        "
export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "
CODE_PATH="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/examples/ziya_llama/finetune_ziya_llama_medical.py"
python3 $CODE_PATH $options
# python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --master_port 40407 $CODE_PATH $options
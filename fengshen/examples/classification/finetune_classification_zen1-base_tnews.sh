#!/bin/bash
#SBATCH --job-name=afqmc-bart-base # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=2 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 
#SBATCH -o %x-%j.log # output and error file name (%x=job name, %j=job id)

export CUDA_VISIBLE_DEVICES='5'
export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions

MODEL_NAME=fengshen-zen1

TASK=tnews
TEXTA_NAME=sentence
LABEL_NAME=label
ID_NAME=id


BATCH_SIZE=8
VAL_BATCH_SIZE=32
ZERO_STAGE=1
STRATEGY=deepspeed_stage_${ZERO_STAGE}

ROOT_DIR=/cognitive_comp/ganruyi/experiments/classification_finetune/${MODEL_NAME}_${TASK}
if [ ! -d ${ROOT_DIR} ];then
  mkdir -p ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

DATA_DIR=/cognitive_comp/yangping/data/ChineseCLUE_DATA/${TASK}_public/
PRETRAINED_MODEL_PATH=/cognitive_comp/ganruyi/hf_models/zen/ZEN_pretrain_base_v0.1.0

CHECKPOINT_PATH=${ROOT_DIR}/ckpt/
OUTPUT_PATH=${ROOT_DIR}/predict.json


config_json="${ROOT_DIR}/ds_config.json"
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
# reduce_bucket_size: hidden_size*hidden_size
# stage3_prefetch_bucket_size: 0.9 * hidden_size * hidden_size
# stage3_param_persistence_threshold: 10 * hidden_size

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $BATCH_SIZE,
  "steps_per_print": 100,
  "gradient_clipping": 0.1,
  "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 2e-5,
      "eps": 1e-12,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params":{
      "warmup_min_lr": 2e-8,
      "warmup_max_lr": 2e-5,
      "warmup_num_steps": 400,
      "warmup_type": "linear"
    }
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


DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.json \
        --valid_data dev.json \
        --test_data test1.1.json \
        --train_batchsize $BATCH_SIZE \
        --valid_batchsize $VAL_BATCH_SIZE \
        --max_length 128 \
        --texta_name $TEXTA_NAME \
        --label_name $LABEL_NAME \
        --id_name $ID_NAME \
        "

MODEL_ARGS="\
        --learning_rate 1e-5 \
        --weight_decay 1e-2 \
        --warmup 0.01 \
        --num_labels 15 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_acc \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 200 \
        --save_weights_only True \
        --dirpath $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_acc:.4f} \
        "


TRAINER_ARGS="\
        --max_epochs 7 \
        --gpus 1 \
        --num_nodes 1 \
        --strategy $STRATEGY \
        --gradient_clip_val 1.0 \
        --check_val_every_n_epoch 1 \
        --val_check_interval 1.0 \
        --default_root_dir $ROOT_DIR \
        "

options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --output_save_path $OUTPUT_PATH \
        --model_type $MODEL_NAME \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif
SCRIPT_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/examples/classification/finetune_classification.py

# python3 $SCRIPT_PATH $options
source activate base
singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH /home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options
# /home/ganruyi/anaconda3/bin/python $SCRIPT_PATH $options


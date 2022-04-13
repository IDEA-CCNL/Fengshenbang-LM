#!/bin/bash
#SBATCH --job-name=slurm-test # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=2 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 


export TORCH_EXTENSIONS_DIR=/cognitive_comp/yangping/cache/torch_extendsions

BERT_NAME=bert-3.9B

TASK=wsc
TEXTA_NAME=texta
LABEL_NAME=label
ID_NAME=id


BATCH_SIZE=16
VAL_BATCH_SIZE=56
ZERO_STAGE=2


ROOT_PATH=cognitive_comp
DATA_DIR=/cognitive_comp/yangping/data/unidata/multichoice/mrc_multichoice_data/other/cluewsc2020/
PRETRAINED_MODEL_PATH=/$ROOT_PATH/yangping/pretrained_model/$BERT_NAME/


CHECKPOINT_PATH=/$ROOT_PATH/yangping/checkpoints/fengshen-finetune/$TASK/
DEFAULT_ROOT_DIR=/cognitive_comp/yangping/nlp/Fengshenbang-LM/fengshen/scripts/log/$TASK
OUTPUT_PATH=/$ROOT_PATH/yangping/nlp/modelevaluation/output/${TASK}_predict.json


config_json="./ds_config.$SLURM_JOBID.json"
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
# reduce_bucket_size: hidden_size*hidden_size
# stage3_prefetch_bucket_size: 0.9 * hidden_size * hidden_size
# stage3_param_persistence_threshold: 10 * hidden_size

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $BATCH_SIZE,
  "steps_per_print": 100,
  "gradient_clipping": 1.0,
  "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 6553600,
        "stage3_prefetch_bucket_size": 5898240,
        "stage3_param_persistence_threshold": 25600,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_fp16_weights_on_model_save": true
    },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-5,
      "betas": [
        0.9,
        0.95
      ],
      "eps": 1e-8,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params":{
      "warmup_min_lr": 5e-6,
      "warmup_max_lr": 1e-5
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
        --test_data test.json \
        --train_batchsize $BATCH_SIZE \
        --valid_batchsize $VAL_BATCH_SIZE \
        --max_length 128 \
        --texta_name $TEXTA_NAME \
        --label_name $LABEL_NAME \
        --id_name $ID_NAME \
        "

MODEL_ARGS="\
        --learning_rate 0.00001 \
        --weight_decay 0.01 \
        --warmup 0.001 \
        --num_labels 2 \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_acc \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 10 \
        --save_weights_only True \
        --dirpath $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_acc:.4f} \
        "
TRAINER_ARGS="\
        --max_epochs 7 \
        --gpus 2 \
        --strategy deepspeed_stage_3 \
        --precision 16 \
        --check_val_every_n_epoch 1 \
        --val_check_interval 10 \
        --default_root_dir $DEFAULT_ROOT_DIR \
        "

options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --output_save_path $OUTPUT_PATH \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

DOCKER_PATH=/$ROOT_PATH/yangping/containers/pytorch21_06_py3_docker_image.sif
SCRIPT_PATH=/$ROOT_PATH/yangping/nlp/fengshen/fengshen/examples/finetune_classification.py

# python3 $SCRIPT_PATH $options
srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $DOCKER_PATH python3 $SCRIPT_PATH $options


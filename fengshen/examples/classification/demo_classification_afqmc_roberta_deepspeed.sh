MODEL_NAME="IDEA-CCNL/Erlangshen-Roberta-110M-NLI"

TEXTA_NAME=sentence1
TEXTB_NAME=sentence2
LABEL_NAME=label
ID_NAME=id

BATCH_SIZE=32
VAL_BATCH_SIZE=32
ZERO_STAGE=1
config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $BATCH_SIZE,
  "steps_per_print": 1000,
  "gradient_clipping": 0.1,
  "zero_optimization": {
        "stage": ${ZERO_STAGE}
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
        --dataset_name IDEA-CCNL/AFQMC \
        --train_batchsize $BATCH_SIZE \
        --valid_batchsize $VAL_BATCH_SIZE \
        --max_length 128 \
        --texta_name $TEXTA_NAME \
        --textb_name $TEXTB_NAME \
        --label_name $LABEL_NAME \
        --id_name $ID_NAME \
        "

MODEL_ARGS="\
        --learning_rate 1e-5 \
        --weight_decay 1e-2 \
        --warmup_ratio 0.01 \
        --num_labels 2 \
        --model_type huggingface-auto \
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor val_acc \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 0 \
        --save_weights_only True \
        --dirpath . \
        --filename model-{epoch:02d}-{val_acc:.4f} \
        "


TRAINER_ARGS="\
        --max_epochs 67 \
        --gpus 1 \
        --num_nodes 1 \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --gradient_clip_val 1.0 \
        --check_val_every_n_epoch 1 \
        --val_check_interval 1.0 \
        --precision 16 \
        --default_root_dir . \
        "

options=" \
        --pretrained_model_path $MODEL_NAME \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "
 
python3 finetune_classification.py $options


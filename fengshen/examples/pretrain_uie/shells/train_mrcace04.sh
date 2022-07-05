#!/bin/bash
DATE=$(date +%F-%H-%M)
task=entity 
dataset=mrc_ace04
model_name=uie-large-en
tokenizer_type=t5sp
epoch=50
batch_size=4
learning_rate=5e-5
warmup_ratio=0.06


DATA_DIR=/cognitive_comp/yangjing/Fengshenbang-LM-data/datasets_eval/text2spotasoc/${task}/${dataset}
PRETRAINED_MODEL_PATH=/cognitive_comp/yangjing/Fengshenbang-LM-data/hf_models/${model_name}
DIR_PATH=/cognitive_comp/yangjing/Fengshenbang-LM-data/log/${task}/${dataset}/${model_name}_b${batch_size}_lr${learning_rate}_wm${warmup_ratio}
mkdir -p ${DIR_PATH}

DATA_ARGS="\
        --train_file ${DATA_DIR}/train.json \
        --valid_file  ${DATA_DIR}/test.json \
        --test_file   ${DATA_DIR}/val.json \
        --record_schema ${DATA_DIR}/record.schema \
        --max_source_length 512 \
        --max_prefix_length -1 \
        --max_target_length 512 \
        --spot_noise 0\
        --asoc_noise 0\
        "


MODEL_ARGS="\
        --learning_rate ${learning_rate} \
        --warmup_ratio ${warmup_ratio} \
        --scheduler_type linear\
        "

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 3 \
        --mode max \
        --save_weights_only true \
        --check_val_every_n_epoch 5 \
        --filename model-{epoch:02d}-{overall-F1:.4f} \
        "


TRAINER_ARGS="\
        --max_epochs ${epoch} \
        --dirpath ${DIR_PATH} \
        "

options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --tokenizer_type ${tokenizer_type} \
        --gpus 1\
        --precision 32\
        --train_batchsize ${batch_size} \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "


SCRIPT_PATH=/cognitive_comp/yangjing/Fengshenbang-LM/fengshen/examples/pretrain_uie/finetune_uie.py

CUDA_VISIBLE_DEVICES='2'
echo -e "Using GPU " ${CUDA_VISIBLE_DEVICES}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /home/yangjing/anaconda3/envs/idea/bin/python  $SCRIPT_PATH $options

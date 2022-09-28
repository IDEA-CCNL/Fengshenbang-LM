#!/bin/bash

MODEL_PATH="/cognitive_comp/zhuojianheng/pretrained_model/wav2vec2-base-ctc-wenet-tencent"
CKPT="/cognitive_comp/zhuojianheng/experiment/fengshen-wav2vec2-base-wenet-ctc-tencent-not-stable-mask-005/ckpt/hf_pretrained_epoch408_step120000"
DATA_HOME="/cognitive_comp/zhuojianheng/data/wenet"
MODEL_NAME="tencent"
LM_PATH="/cognitive_comp/zhuojianheng/data/wenet/lm_model/test.bin"
# LM_PATH是kenlm模型的路径

for DATA_SET in dev
do
    python3 inference.py \
    --model_path $MODEL_PATH \
    --ckpt $CKPT \
    --lm_path $LM_PATH \
    --tsv ${DATA_HOME}/${DATA_SET}/data.tsv \
    --wrd ${DATA_HOME}/${DATA_SET}/data.wrd \
    --target ${MODEL_NAME}_${DATA_SET}.tem 
done


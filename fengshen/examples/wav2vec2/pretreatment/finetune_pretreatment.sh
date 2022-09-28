#!/bin/bash
MANIFEST_PATH="/cognitive_comp/zhuojianheng/data/wenet/S"
OUTPUT_DIR="/cognitive_comp/zhuojianheng/data/wenet/S"
TOKENIZER_DIR="/cognitive_comp/zhuojianheng/pretrained_model/wav2vec2-base-ctc-wenet"

python wenet_labels.py --output_dir $OUTPUT_DIR --tsv ${MANIFEST_PATH}/train.tsv --output_name train --tokenizer_path $TOKENIZER_DIR
# 如果指定了tokenizer_path，程序会从语料中构造一个tokenizer，并保存在tokenizer_path，一般设置为模型config所在的目录
python wenet_labels.py --output_dir $OUTPUT_DIR --tsv ${MANIFEST_PATH}/valid.tsv --output_name valid

#!/bin/bash
set -x -e

echo "START TIME: $(date)"
BIN_DIR=/cognitive_comp/ganruyi/experiments/randeng_t5_char_57M/randeng_t5_char_57M
if [ ! -d ${BIN_DIR} ];then
  mkdir ${BIN_DIR}
  echo ${BIN_DIR} created!!!!!!!!!!!!!!
else
  echo ${BIN_DIR} exist!!!!!!!!!!!!!!!
fi

export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions


MODEL_ARGS="
    --ckpt_path /cognitive_comp/ganruyi/experiments/randeng_t5_char_57M/ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt \
    --bin_path ${BIN_DIR}/pytorch_model.bin \
    --rm_prefix module.model. \
"

SCRIPTS_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/examples/pretrain_t5/convert_ckpt_to_bin.py

export CMD=" \
    $SCRIPTS_PATH \
    $MODEL_ARGS \
    "

echo $CMD
/home/ganruyi/anaconda3/bin/python $CMD

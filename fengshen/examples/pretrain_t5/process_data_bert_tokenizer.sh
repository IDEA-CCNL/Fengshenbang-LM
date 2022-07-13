#!/bin/bash
#SBATCH --job-name=process_data_bert_tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # number of gpus
#SBATCH --cpus-per-task=120 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o /cognitive_comp/ganruyi/experiments/randeng_t5_char_77M/%x-%j.log
#SBATCH -e /cognitive_comp/ganruyi/experiments/randeng_t5_char_77M/%x-%j.err
set -x -e

echo "START TIME: $(date)"

DATA_ARGS="
    --tokenizer_type bert_tokenizer \
    --train_data_path wudao_180g \
    --train_split_size 0.999 \
    --max_seq_length 512 \
    --preprocessing_num_workers 100 \
    --saved_data_shards 800 \
    --saved_train_data_path /cognitive_comp/common_data/wudao_180g_bert_tokenized_512_train/ \
    --saved_test_data_path /cognitive_comp/common_data/wudao_180g_bert_tokenized_512_test/ \
    --pretrained_model_path /cognitive_comp/ganruyi/experiments/randeng_t5_char_77M/randeng_t5_char_77M \
    --text_column_name text \
    --remove_columns token_type_ids text \
"

    # --remove_columns text \
SCRIPTS_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/examples/pretrain_t5/process_data.py

export CMD=" \
    $SCRIPTS_PATH \
    $DATA_ARGS \
    "

echo $CMD
source activate base
/home/ganruyi/anaconda3/bin/python $CMD
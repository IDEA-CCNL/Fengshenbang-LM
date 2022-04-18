#!/bin/bash
#SBATCH --job-name=google_sp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --ntasks-per-node=1
#SBATCH -o %x-%j.log

set -x -e

echo "START TIME: $(date)"

BIN_PATH=/cognitive_comp/gaoxinyu/sentencepiece/sentencepiece/bin/usr/local/bin/spm_train
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/cognitive_comp/gaoxinyu/sentencepiece/sentencepiece/bin/usr/local/lib
INPUT_FILE=/cognitive_comp/gaoxinyu/github/Fengshenbang-LM/fengshen/tokenizer/sentencepiece/shuffle_corpus_59132213.txt
INPUT_FILE_SMALL=/cognitive_comp/gaoxinyu/github/Fengshenbang-LM/fengshen/tokenizer/sentencepiece/shuffle_corpus_1000000.txt


VOCAB_SIZE=40000
COV=0.9995
MAX_LENGTH=6
TYPE=bpe
SEED=42
MAX_INPUT_LENGTH=100000

OPTION="\
    --input=${INPUT_FILE} \
    --vocab_size=${VOCAB_SIZE} \
    --character_coverage=${COV} \
    --max_sentencepiece_length=${MAX_LENGTH} \
    --model_type=${TYPE} \
    --model_prefix=${TYPE}_v${VOCAB_SIZE}_s${SEED}_cov${COV}_max${MAX_LENGTH} \
    --random_seed=${SEED} \
    --max_sentence_length=100000 \
    --shuffle_input_sentence=true \
    --input_sentence_size=${MAX_INPUT_LENGTH} \
    --minloglevel 1 \
    --num_threads=100 \
    --train_extremely_large_corpus=true \
    "

eval $BIN_PATH $OPTION
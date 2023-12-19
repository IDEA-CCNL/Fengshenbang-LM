#!/bin/bash

#SBATCH --job-name=medical-gpt # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 -p pol # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)

# export CUDA_VISIBLE_DEVICES='0'
ARGS="\
        --prefix medical-gpt \
        --ckpt /cognitive_comp/ganruyi/huggingface_models/ziya-llama-13b-medical-merged \
        --sample_num 15 \
        "

python -u predict_medical_qa.py $ARGS
#!/bin/bash

#SBATCH --job-name=chimed-gpt # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=20 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 -p pol # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)

ARGS="\
        --prefix chimed-gpt \
        --ckpt /cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/baseline_ckpt/hf_pretrained_epoch0_step1400 \
        --sample_num 15 \
        "

python predict_medical_qa.py $ARGS
#!/bin/bash

#SBATCH --job-name=gpt-4 # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G # memory per cpu-core (4G is default)
#SBATCH -p pol # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)

ARGS="\
        --prefix gpt-4 \
        --sample_num 2 \
        "

python predict_medical_qa.py $ARGS
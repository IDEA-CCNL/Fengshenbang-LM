#!/bin/bash
#SBATCH --job-name=evaluate_model # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # number of tasks to run per node
#SBATCH --cpus-per-task=5 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH -o inference_log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -p batch
#SBATCH --qos=ai4cogsys

export SCRIPT_PATH=./evaluate_model.py 

MODEL_PATH=''

srun python $SCRIPT_PATH $MODEL_PATH
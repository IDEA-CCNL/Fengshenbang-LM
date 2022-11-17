#!/bin/bash
#SBATCH --job-name=evaluate_model # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # number of tasks to run per node
#SBATCH --cpus-per-task=5 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH -o inference_log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -p batch
#SBATCH --qos=ai4cogsys

export SCRIPT_PATH=/cognitive_comp/lixiayu/work/Fengshenbang-LM/fengshen/examples/finetune_taiyi_stable_diffusion/evaluate_model.py

srun /home/lixiayu/anaconda3/envs/stable/bin/python $SCRIPT_PATH
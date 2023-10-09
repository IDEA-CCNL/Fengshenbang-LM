#!/bin/bash

#SBATCH --job-name=debug # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=10 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 -p pol # number of gpus per node

cat /etc/hosts
python -c 'import torch; print(torch.__version__); print(torch.zeros(10,10).cuda().shape)'
# jupyter lab --ip=0.0.0.0 --port=8888

/home/ganruyi/code-server/code-server-4.10.0-linux-amd64/bin/code-server --host 0.0.0.0 --port 40405
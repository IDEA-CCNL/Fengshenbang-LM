#!/bin/bash
#SBATCH --job-name=ziya-v1 # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=10 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pol-preempted # number of gpus per node preempted
#SBATCH -o chimed-%x-%j.log

MODEL_PATH=/cognitive_comp/yangping/checkpoints/llama/llama2hf/hf_llama13b_step43000
# MODEL_PATH=/cognitive_comp/common_checkpoint/llama2_hf_13b_step191000
# MODEL_PATH=/cognitive_comp/ganruyi/huggingface_models/baichuan/Baichuan-13B-Base
# MODEL_PATH=/cognitive_comp/ganruyi/huggingface_models/bentsao-7b
# MODEL_PATH=/cognitive_comp/ganruyi/huggingface_models/medicalgpt-ziya-13b
# MODEL_PATH=/cognitive_comp/ganruyi/huggingface_models/medicalgpt-baichuan-13b
# MODEL_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/output/checkpoint-115000


DEV_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/chimed.dev.json
MODEL_NAME="ziya-v1"
TASK="chimed"
SAVE_PATH=./${TASK}_${MODEL_NAME}_predict.json
echo $SAVE_PATH

python /cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/generate.py \
--task $TASK \
--dev $DEV_PATH \
--save_path $SAVE_PATH \
--model_path $MODEL_PATH \
--model_name $MODEL_NAME \
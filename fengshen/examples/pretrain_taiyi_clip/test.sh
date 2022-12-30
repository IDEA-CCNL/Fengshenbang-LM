#!/bin/bash
#SBATCH --job-name=finetune_taiyi # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

# pwd=Fengshenbang-LM/fengshen/examples/pretrain_erlangshen

NNODES=1
GPUS_PER_NODE=1

MICRO_BATCH_SIZE=64

DATA_ARGS="\
        --test_batchsize $MICRO_BATCH_SIZE  \
        --datasets_name flickr30k-CNA \
        "

MODEL_ARGS="\
        --model_path /cognitive_comp/gaoxinyu/github/Fengshenbang-LM/fengshen/workspace/taiyi-clip-huge-v2/hf_out_0_661 \
        "

TRAINER_ARGS="\
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy ddp \
        --log_every_n_steps 0 \
        --default_root_dir . \
        --precision 32 \
        "
# num_sanity_val_steps， limit_val_batches 通过这俩参数把validation关了

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $TRAINER_ARGS \
        "

CUDA_VISIBLE_DEVICES=0 python3 test.py $options
#srun -N $NNODES --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE --cpus-per-task=20 python3 pretrain.py $options

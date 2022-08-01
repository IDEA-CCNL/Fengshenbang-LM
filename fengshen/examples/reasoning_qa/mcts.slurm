#!/bin/bash
  
#SBATCH -J mcts
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-gpu=32
#SBATCH --gres=gpu:1
#SBATCH -x dgx[045,050,051,073]
#SBATCH -o logs/mcts-%j.log
#SBATCH -e errs/mcts-%j.err

NUM_NODES=1
NUM_GPUS=1

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"
ROOT_DIR=/cognitive_comp/zhuxinyu/codes/reasoning_qa


MCTS_ARGS="
    --max_num_children 4 \
    --root_max_num_children 10 \
    --roll_out_size 300 \
    --sampling_size 50 \
    --max_length 400 \
    --max_iter 100 \
    --sim_score_base 0.6 \
    --time_out 240 \
    --alpha 1. \
    --c_out 10 \
    --burn_in_rate 0.2 \
    --bp_decay_rate 1.0 \
    --annealing_rate 10 \
    --rep_penalty 0.0 \
    --ng 2 \
    --sample_capacity 1000 \
    --temperature 0.7 \
    --data /cognitive_comp/zhuxinyu/datasets/grade-school-math/grade_school_math/data/test_with_qid_split0.jsonl \
    --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/hf_pretrained_epoch1_step936-GSM-07-26_18-35-44-upsampling_10_times_train_data_with_above_avg_score_plus_0.1_correct_solution_version_0.2.1_with_mcts_samples_loss_on_prefix/hf_pretrained_epoch0_step1198 \
    --verifier_type deberta \
    --verifier_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/deberta-v3-large-Verifier-GSM-06-16_11-55-56-verifier_head_logits/hf_pretrained_epoch2_step17517 \
    --expand_verifier_type gpt \
    --expand_verifier_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/init_from_gpt-j-6B-GSM-05-10_12-33_hf_pretrained_epoch1_74w_data_for_train/hf_pretrained_epoch0_step46706 \
    --timestamp $TIMESTAMP \
    --expand_length 20 \
    --split 0 \
    --expand_repeat_penalty 1.2 \
"
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch0_step468 \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-05-17_13-13-39_loss_on_prefix_with_collate_fn/hf_pretrained_epoch9_step2340 \
    # --expand_verifier_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/init_from_gpt2-large-GSM-05-13_12-57-03_hf_pretrained_epoch13/hf_pretrained_epoch0_step22275 \
    # --data /cognitive_comp/zhuxinyu/datasets/grade-school-math/grade_school_math/data/bug_test.jsonl \
    # --verifier_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/init_from_gpt2-large-GSM-05-13_12-57-03_hf_pretrained_epoch13/hf_pretrained_epoch0_step22275 \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-05-17_13-13-39_loss_on_prefix_with_collate_fn/hf_pretrained_epoch9_step2340 \
    # --data /cognitive_comp/zhuxinyu/datasets/grade-school-math/grade_school_math/data/1-440_test.jsonl \
    # --data /cognitive_comp/zhuxinyu/datasets/grade-school-math/grade_school_math/data/441-880_test.jsonl \
    # --model_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-05-17_13-13-39_loss_on_prefix_with_collate_fn/hf_pretrained_epoch9_step2340 \
    # --verifier_type bert \
    # --verifier_name /cognitive_comp/zhuxinyu/codes/reasoning_qa/verifier_outputs/bert-large-uncased-Verifier-GSM-05-30_23-39-11/hf_pretrained_epoch1_step22276 \
    # --initialize_tree \
    # --use_cls

SCRIPTS_PATH=$ROOT_DIR/mcts.py

export CMD=" \
    $SCRIPTS_PATH \
    $MCTS_ARGS \
    "

echo $CMD
bash -c 'python $CMD'
echo "END TIME: $(date)"


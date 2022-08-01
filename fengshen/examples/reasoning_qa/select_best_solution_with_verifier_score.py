# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : select_best_solution_with_verifier_score.py
#   Last Modified : 2022-05-18 11:30
#   Describe      : 
#
# ====================================================

from data_preprocess import DataProcessor
import re


ANS_RE = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)\<\|endoftext\|\>")
old_template = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)$")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if not match:
        match = old_template.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            match_str = float(match_str)
            match_str = round(match_str, 3)
            match_str = str(match_str)
        except:
            print("matched but not a float", match_str)
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS


if __name__ == "__main__":
    #  file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-05-17_13-13-39_loss_on_prefix_with_collate_fn/hf_pretrained_epoch9_step2340/06-07_17-25-29-mcts_verifier_file.jsonl"
    #  file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936/06-21-mcts_verifier_file/merged-06-21-mcts_verifier_file-gpt-j.jsonl_voting_acc_4132_verifier_acc_4230_verifier_scored_0"
    file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936/mawps-07-31_23-12-14-mcts_verifier_file.jsonl"
    data = DataProcessor._read_jsonl(file_name)
    qid_set = set()
    #  best_solution = [{"verifier_score": -float('inf'), "question_id": x, "is_correct": False} for x in range(1319)]
    best_solution = [{"verifier_score": -float('inf'), "question_id": x, "is_correct": False} for x in range(7473)]
    for ex in data:
        qid_set.add(int(ex['question_id']))
        if float(ex['verifier_score']) > float(best_solution[int(ex['question_id'])]['verifier_score']):
            best_solution[int(ex['question_id'])]['solution'] = ex['solution']
            best_solution[int(ex['question_id'])]['verifier_score'] = ex['verifier_score']
            #  best_solution[int(ex['question_id'])]['is_correct'] = ex['is_correct']
            best_solution[int(ex['question_id'])]['is_correct'] = extract_answer(ex['solution']) == extract_answer(ex['ground_truth'])
    correct = 0
    for s in best_solution:
        if s['is_correct']:
            correct += 1
    acc = correct / len(qid_set)
    print("Verifier Accuracy: ", acc)

    for ex in data:
        if ex['is_correct']:
            best_solution[int(ex['question_id'])]['is_correct'] = True
    correct = 0
    for s in best_solution:
        if s['is_correct']:
            correct += 1
    acc = correct / len(qid_set)
    print("Test@100 Accuracy / Solve Rate: ", acc)


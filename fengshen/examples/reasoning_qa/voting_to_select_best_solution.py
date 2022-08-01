# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : voting_to_select_best_solution.py
#   Last Modified : 2022-06-13 17:25
#   Describe      : 
#
# ====================================================
from pysnooper import snoop
from data_preprocess import DataProcessor
#  from math_data_model import extract_answer
import re
from collections import defaultdict

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

#  @snoop()
def main():
    #  file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-05-17_13-13-39_loss_on_prefix_with_collate_fn/hf_pretrained_epoch9_step2340/merged_05-17_22-39-06-generator_solution.jsonlprocessed_gpt-j-epoch0_verifier_scored"
    #  file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936/07-11-mcts_verifier_file_capacity_1000_trainset/merged_07-11_12_13-mcts_verifier_file.jsonl_verifier_acc_7424_voting_acc_7322_solve_rate_8541"
    file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936/mawps-07-31_23-12-14-mcts_verifier_file.jsonl"
    data = DataProcessor._read_jsonl(file_name)
    #  all_q_voting = [defaultdict(int) for i in range(1319)]
    all_q_voting = [defaultdict(int) for i in range(7473)]
    gt_answer_dict = dict()
    qid_set = set()
    for idx, ex in enumerate(data):
        qid = int(ex['question_id'])
        qid_set.add(qid)
        pd_answer = extract_answer(ex['solution'])
        if pd_answer != INVALID_ANS:
            all_q_voting[qid][str(pd_answer)] += float(ex['verifier_score'])
        gt_answer = extract_answer(ex["ground_truth"])
        gt_answer_dict[str(qid)] = gt_answer

    correct = 0
    for idx, voting in enumerate(all_q_voting):
        if len(voting) == 0:
            continue
        pd_answer = max(voting, key=voting.get)
        if pd_answer == gt_answer_dict[str(idx)]:
            correct += 1
    acc = correct / len(qid_set)
    print(len(qid_set))
    print("Voting Accuracy: ", acc)

if __name__ == "__main__":
    main()


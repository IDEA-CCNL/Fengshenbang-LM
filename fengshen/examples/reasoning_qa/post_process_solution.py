# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : post_process_solution.py
#   Last Modified : 2022-05-09 15:56
#   Describe      : 
#
# ====================================================
from data_preprocess import DataProcessor
import re
import jsonlines
import copy
#  from math_data_model import extract_answer, INVALID_ANS


if __name__ == "__main__":
    file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt2-large-GSM-05-17_13-13-39_loss_on_prefix_with_collate_fn/hf_pretrained_epoch1_step468/merged_05-17_18-46-16-generator_solution.jsonl"
    data = DataProcessor._read_jsonl(file_name)
    #  pat = re.compile("#### [0-9,.-]*\<\|endoftext\|\>")
    #  ANS_RE = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)")
    #  pat_without_end = re.compile("#### [0-9,.-]*")
    invalid_solution_count = 0
    valid_solution_count = 0
    pat_without_end = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)")
    with jsonlines.open(file_name + "processed", "w") as f:
        for d in data:
            if "new_iteration" in d:
                continue
            #  processed_solution = copy.deepcopy(d)
            d['ground_truth'] = d['ground_truth'].replace("<|endoftext|>", "").strip()
            #  if extract_answer(d['solution']) != INVALID_ANS:
            match = pat_without_end.search(d['solution'])
            if match:
                d['solution'] = d['solution'][:match.end()].replace("<|endoftext|>", "").strip()
                valid_solution_count += 1
            else:
                invalid_solution_count += 1
                print(d['solution'])

            f.write(d)

    print("Total valid solution count: ", valid_solution_count)
    print("Total invalid solution count: ", invalid_solution_count)


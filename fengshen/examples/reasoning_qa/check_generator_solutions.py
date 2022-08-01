# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : check_generator_solutions.py
#   Last Modified : 2022-05-10 21:21
#   Describe      : 
#
# ====================================================
from data_preprocess import DataProcessor
import re
import jsonlines
import copy
#  from math_data_model import extract_answer, INVALID_ANS


if __name__ == "__main__":
    file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-05-10_12-33/hf_pretrained_epoch1_step936/05-10_18-38-generator_solution.jsonl"
    #  file_name = "/cognitive_comp/zhuxinyu/codes/reasoning_qa/outputs/gpt-j-6B-GSM-04-18_15-45/hf_pretrained_epoch0_step0/05-09_14-42-model_solution_skip_special_tokens_when_decoder.jsonl"
    invalid_solution_count = 0
    valid_solution_count = 0
    #  pat_without_end = re.compile(r"\[ANS\] \-?[0-9\.\,]+\<\|endoftext\|\>")
    pat_without_end = re.compile(r"\[ANS\] \-?[0-9\.\,]+")
    pat_without_end_old = re.compile(r"#### \-?[0-9\.\,]+\<\|endoftext\|\>")
    #  pat_without_end_old = re.compile(r"#### \-?[0-9\.\,]+")
    for idx in range(4):
        data = DataProcessor._read_jsonl(file_name + str(idx))
        #  pat = re.compile("#### [0-9,.-]*\<\|endoftext\|\>")
        #  ANS_RE = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)")
        for d in data:
            match = pat_without_end.search(d['solution']) or pat_without_end_old.search(d['solution'])
            if match is None:
                if "[ANS]" in d['solution']:
                    print(d['solution'])
                    print(match)
                invalid_solution_count += 1
            else:
                #  print(match)
                valid_solution_count += 1
    print("Total valid solution count: ", valid_solution_count)
    print("Total invalid solution count: ", invalid_solution_count)


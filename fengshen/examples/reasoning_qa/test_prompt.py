# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : test_prompt.py
#   Last Modified : 2022-04-28 15:43
#   Describe      : 
#
# ====================================================
from data_preprocess import DataProcessor

a = DataProcessor._read_jsonl("/cognitive_comp/zhuxinyu/datasets/chain-of-thought-prompting/modified_prompt.jsonl")

t = DataProcessor._read_jsonl("/cognitive_comp/zhuxinyu/datasets/grade-school-math/grade_school_math/data/test.jsonl")
t0 = t[0]
print(a[0])
print(t0)

print(a[0]['prompt'])
print(t0['answer'])


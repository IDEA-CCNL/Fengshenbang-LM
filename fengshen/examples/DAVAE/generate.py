# -*- encoding: utf-8 -*-
'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   generate.py
@Time    :   2022/11/04 19:17
@Author  :   Liang Yuxin
@Version :   1.0
@Contact :   liangyuxin@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''
# here put the import lib

import torch
from fengshen.models.DAVAE.DAVAEModel import DAVAEModel
from transformers import BertTokenizer,T5Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese")
decoder_tokenizer = T5Tokenizer.from_pretrained("IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese", eos_token = '<|endoftext|>', pad_token = '<pad>',extra_ids=0)
decoder_tokenizer.add_special_tokens({'bos_token':'<bos>'})
vae_model = DAVAEModel.from_pretrained("IDEA-CCNL/Randeng-DAVAE-1.2B-General-Chinese").to(device)
input_texts = [
    "针对电力系统中的混沌振荡对整个互联电网的危害问题,提出了一种基于非线性光滑函数的滑模控制方法.",
    "超市面积不算大.挺方便附近的居民购买的. 生活用品也比较齐全.价格适用中.",
]
output_texts = vae_model.simulate_batch(encoder_tokenizer,decoder_tokenizer,input_texts)
print(output_texts)

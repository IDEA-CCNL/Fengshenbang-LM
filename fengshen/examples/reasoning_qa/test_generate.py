# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : test_generate.py
#   Last Modified : 2022-04-29 17:07
#   Describe      : 
#
# ====================================================
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTJForCausalLM
import torch as th
import random
from torchsnooper import snoop
import time


#  @snoop()
def sample(model, qn, tokenizer, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    trigger_tokens = set([800, 198, 628])
    device = "cuda"

    generated_token_ids = []
    past_key_values = None
    model_kwargs = {}
    toks = tokenizer([qn], padding=False, return_tensors="pt").input_ids.to(device)
    for _ in range(sample_len):
        with th.no_grad():
            #  toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
            #  orig_len = toks["input_ids"].shape[1]
            orig_len = toks.size(1)

            model_kwargs["past"] = past_key_values
            out, model_outputs = model.generate(
                #  **toks, max_length=orig_len + 1, pad_token_id=tokenizer.pad_token_id
                toks, max_length=orig_len + 1, 
                pad_token_id=model.config.eos_token_id, 
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True,
                output_scores=True,
                use_cache=True,
                **model_kwargs,
            )
            # ([past_key_layer0, past_value_layer0], [past_key_layer1, past_value_layer1], ..., [past_key_layer11, past_value_layer11])
            # len(past_key_values) = num_layers, len(past_key_values[0]) = 2
            # past_key_values[0][0].size() = (batch_size, num_heads, seq_len, hidden_size // num_heads)
            past_key_values = model_outputs.past_key_values
            #  模拟calculator接管的情况 
            #  if random.random() > 0.5:
            #      out.sequences[0, -1] = 800
            if out.sequences[0, -1].item() in trigger_tokens:
                out.sequences[0, -1] = 11

            generated_token_ids.append(out.sequences[0][-1])
            toks = out.sequences
            #  text = tokenizer.batch_decode(out.sequences)[0]

            #      text = text + "10" + ">>"
            #      generated_token_ids.extend(tokenizer.convert_tokens_to_ids(["10", ">>"]))
            #      past_key_values = None

            #  qn = text

    return qn, generated_token_ids


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("/cognitive_comp/zhuxinyu/pretrained_models/gpt2/")
    tokenizer = GPT2Tokenizer.from_pretrained("/cognitive_comp/zhuxinyu/pretrained_models/gpt2/")
    model.cuda()
    qn = "A thought: my motivations for writing this letter. One dream I have as a graduate student is to be a good graduate student mentor. I think an important part to being a good mentor for undergraduates is having the ability to understand undergraduate students. I recently finished my undergraduate studies and before time dilutes my memory of my undergraduate research experiences, I decided to write a blog post about what I learned from those experiences. Diving into research (an entirely different career than what I considered doing) and computer science (an entirely different major than what I considered doing) was scary for me because of the uncertainty. Will I be good at research? Will my project turn into anything? What will the future"
    start_time = time.time()
    qn, generated_token_ids = sample(model, qn, tokenizer, 100)
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    print(qn)
    print(tokenizer.decode(generated_token_ids))


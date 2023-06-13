"""
这是基于hugging face社区开源的框架accelerate制定的基础量化推理方案
该框架主要实现了int8、int4量化，以及cpu或者disk offload
实现了用低存储，小设备运行大模型
具体可以见wiki：http://wiki.team.idea.edu.cn/pages/viewpage.action?pageId=31464125
"""
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt
import torch


# 量化的方案集成到from_pretrained方法中了
# 如果要量化加载，device_map必须设置
# 量化的参数主要是：load_in_8bit，load_in_4bit (最新的main分支有文档说明，transformer4.29.2还没有4bit)
# 更多参考文档：https://huggingface.co/docs/accelerate/usage_guides/big_modeling


def load_model_source(model_path, load_in_8bit=True):
    if load_in_8bit:
        lm = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', load_in_8bit=load_in_8bit).eval()
    else:
        lm = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto',torch_dtype=torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 查看加载后的模型，所占内存
    print(f'模型所占显存: {lm.get_memory_footprint()/1024/1024/1024} GB')
    # 查看模型的分布
    print('模型在设备上的分布：\n', lm.hf_device_map)
    return lm, tokenizer


def decode_speed_test(lm, tokenizer, batch_size=1, generate_lenght=100, test_round=5):
    """
    测试推理速度
    """
    st = time.time()
    text = ['中国的首都是'] * batch_size
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(0)
    for _ in range(test_round):
        out = lm.generate(input_ids, max_new_tokens=generate_lenght)
    time_cost = time.time()-st
    total_token_gen = batch_size*generate_lenght*test_round
    token_gen_speed = total_token_gen/time_cost
    per_token_time_cost = time_cost/total_token_gen*1000
    info = f"""
    bs:{batch_size} max_new_tokes:{generate_lenght} test_round:{test_round}
    generate total token: {total_token_gen} sec
    speed: {token_gen_speed:.2f} token/sec
    token_time_cost: {per_token_time_cost:.2f} ms
    """
    print(info)
    return out, info


def generate(text, max_new_tokens=128, do_sample=True, top_p=0.9, return_n=5):
    text = f'<human>:{text.strip()}\n<bot>:'
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(0)
    out = lm.generate(input_ids,
                      max_new_tokens=max_new_tokens,
                      do_sample=do_sample,
                      top_p=top_p,
                      num_return_sequences=return_n)
    seq = tokenizer.batch_decode(out)
    return out, seq


if __name__ == '__main__':
    model_path = '/cognitive_comp/common_data/Huggingface-Models/IDEA-CCNL/Ziya-LLaMA-13B-RLHF-V1'
    lm, tokenizer = load_model_source(model_path)
    # _, _ = decode_speed_test(lm, tokenizer)
    _,seq = generate('中国的首都是哪里？')

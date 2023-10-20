from time import time
from random import choices
from vllm import LLM, SamplingParams

st = time()
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/baseline_ckpt_old/hf_pretrained_epoch2_step4716'
# path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/baseline_ckpt/hf_pretrained_epoch1_step3138'
# path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/baseline_ckpt/hf_pretrained_epoch2_step4707'
path = '/cognitive_comp/common_checkpoint/llama2_hf_13b_step191000'
llm = LLM(model=path,tensor_parallel_size=2)
print(f'load model cost {time()-st:.5f} secend')
prompts = [
    '问题:写一个童话故事，名字叫《长颈鹿找工作》\n回答:',
]

sampling_params = SamplingParams(temperature=0.85, top_p=0.85,max_tokens=1024)

# # 预热
# outputs = llm.generate(prompts[:5], sampling_params)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

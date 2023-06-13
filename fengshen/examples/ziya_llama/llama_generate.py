
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from typing import List
import torch.nn.functional as F

def zero_pad_sequences(sequences: List[torch.Tensor], side: str = 'left', padding_value: int = 0) -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=padding_value))
    return torch.stack(padded_sequences, dim=0)

def generate(queries: List[str], tokenizer: AutoTokenizer, model: LlamaForCausalLM, device: int=0, **generate_kwargs):
    def _apply_prefix(query):
        return f"<human>:{query.strip()}\n<bot>:"

    def _tokenizing(queries):

        input_ids = []
        for query in queries:
            query = _apply_prefix(query)
            input_ids.append(torch.tensor(tokenizer(query).input_ids))
        inputs = zero_pad_sequences(input_ids, side="left", padding_value=generate_kwargs["pad_token_id"])
        return inputs


    input_ids = _tokenizing(queries).to(device)
    pad_token_id = generate_kwargs["pad_token_id"]
    input_attention_mask = input_ids.not_equal(pad_token_id).to(dtype=torch.bool, device=device)
    sequences = model.generate(
        input_ids.to(device), attention_mask=input_attention_mask, **generate_kwargs)
    output = []
    for seq in sequences:
        out_text = tokenizer.decode(seq.tolist(), skip_special_tokens=False).split('<bot>:')[-1]
        output.append(out_text.replace('<s>','').replace('</s>',''))
    return output

if __name__ == '__main__':
	model_path = '/cognitive_comp/wanghao/models/llama_sft/llama_stage_step1900_ppo_step49_hf'
	tk_path = '/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/'

	model = LlamaForCausalLM.from_pretrained(model_path).to(torch.bfloat16).cuda()
	llama_tokenizer = AutoTokenizer.from_pretrained(tk_path)

	generate_kwargs = {
        	"do_sample": True,
        	"top_p": 1.0,   
        	"top_k": 0,
        	"max_length": 2048,
        	"repetition_penalty": 1.0,
        	"temperature": 0.8,
        	"pad_token_id": llama_tokenizer.eos_token_id,
        	"eos_token_id": llama_tokenizer.eos_token_id,
	}

	queries = ['怎样给世界一点爱？', '生命的意义是什么？']
	ans = generate(queries=queries,
         	tokenizer=llama_tokenizer,
         	model=model,
         	device=0,
         	**generate_kwargs)

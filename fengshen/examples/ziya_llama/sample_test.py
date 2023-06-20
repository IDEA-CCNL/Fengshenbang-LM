from transformers import LlamaTokenizer
import torch


def pad(ids, pad_id, max_length):
    if len(ids) > max_length:
        return ids[:max_length]
    return ids + [pad_id] * (max_length - len(ids))

prompt_without_output = "<human>:{prompt}\n<bot>:"

def generate_samples(s, tokenizer, max_seq_length):
    max_length = 0
    prompt_cnt = min(len(s["prompt"]), len(s["output"]))
    input_ids_list = []
    labels_list = []
    input_ids = []
    labels_ids = []
    for i in range(prompt_cnt):
        prompt_input_ids = tokenizer(prompt_without_output.format_map(
            {"prompt": s["prompt"][i].strip()}), add_special_tokens=False).input_ids
        output_ids = tokenizer(s["output"][i].strip(), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        input_ids += prompt_input_ids
        input_ids += output_ids
        labels_ids += [-100] * (len(prompt_input_ids)) + output_ids
    
    # input_ids += [self.tokenizer.eos_token_id]
    # labels_ids += [self.tokenizer.eos_token_id]
    max_length = min(max(len(input_ids), max_length), max_seq_length)
    input_ids_list.append(input_ids)
    labels_list.append(labels_ids)
    
    # PAD
    for i in range(len(input_ids_list)):
        labels_list[i] = pad(labels_list[i], -100, max_length)
        input_ids_list[i] = pad(input_ids_list[i], tokenizer.pad_token_id, max_length)
    model_inputs = {
        'input_ids': torch.tensor(input_ids_list).clone(),
        'attention_mask': torch.ones((len(input_ids_list), max_length)).clone(),
        "position_ids": torch.arange(0, max_length).unsqueeze(0).expand(len(input_ids_list), max_length).clone(),
        'labels': torch.tensor(labels_list).clone(),
    }

    return model_inputs


if __name__ == "__main__":
    tokenizer = LlamaTokenizer.from_pretrained("/cognitive_comp/gaoxinyu/workspace_lightning/llama/ckpt/GXY_HIT_13B")
    s = {'task': 'belle_multi_chat', 
         'prompt': ['写一篇关于人工智能对未来影响的文章，2000字以上。', '从这篇文章中提取出未来人工智能发展方向的关键词。'], 
         'output': ['人工。', '未来 人工智能 智能 可靠 透明 应用 领域 利用 有益助手']}
    batch = generate_samples(s, tokenizer, 256)
    def detokenize(token_ids):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        return tokenizer.convert_tokens_to_string(tokens)

    print('source: {}'.format(batch['input_ids']))
    print('target: {}'.format(batch['labels']))
    print('source: {}'.format(detokenize(batch['input_ids'])))
    label_idx = batch['labels'][1] != -100
    print('target: {}'.format(detokenize(
        batch['labels'][0][label_idx])))
    print('mask: {}'.format(batch['attention_mask'][1]))
    print('position_ids: {}'.format(batch['position_ids'][1]))
    
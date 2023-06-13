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
         'output': ['人工智能是一种非常热门的技术，正在改变我们生活的方方面面。人工智能可应用于医疗、金融、物流、无人驾驶等领域，带来了极大的便利和效益。然而，这种技术的快速发展也引发了许多人对于其影响的担忧，其中最主要的担忧是人工智能是否会取代人类工作。虽然无法预测未来，但如果我们能够善加利用人工智能，他们就有可能成为我们生活和工作的有益助手。在这篇文章里，我将会详细探讨人工智能对未来的一些可能影响，以及我们可以怎样利用它们，而不是让人工智能取代人类。\n人工智能在医疗领域的应用已经开始改善了很多人的生活。例如，利用深度学习的技术，可以做出更准确的诊断，使我们为治疗患者提供更好的治疗方案。在金融领域，人工智能已经被广泛应用，例如在诈骗检测和风险管理方面。物流公司也使用人工智能来提高能源效率和优化配送路线。另外，在无人驾驶汽车和机器人领域，人工智能技术的应用是不可避免的。\n然而，人工智能也会带来某些负面的影响。例如，自动化将会替代一些人的工作，对其收入和安全造成威胁。另外，人工智能也可能带来新的安全隐患，例如黑客可以利用漏洞入侵我们的系统。因此，我们需要建立可靠的人工智能系统，以避免这些潜在的安全问题。\n未来，人工智能的发展方向将会是建立更为智能、可靠和透明的系统。我们需要在确保安全的前提下，与人工智能一同工作，并将其应用于更多的领域。如果我们能够善加利用人工智能，它们将会成为我们生活和工作中不可或缺的有益助手。', '未来 人工智能 智能 可靠 透明 应用 领域 利用 有益助手']}
    batch = generate_samples(s, tokenizer, 256)
    def detokenize(token_ids):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        return tokenizer.convert_tokens_to_string(tokens)

    print('source: {}'.format(batch['input_ids'][0]))
    print('target: {}'.format(batch['labels'][0]))
    print('source: {}'.format(detokenize(batch['input_ids'][0])))
    label_idx = batch['labels'][0] != -100
    print('target: {}'.format(detokenize(
        batch['labels'][0][label_idx])))
    print('mask: {}'.format(batch['attention_mask'][0]))
    print('position_ids: {}'.format(batch['position_ids'][0]))
    
import torch
import torch.nn.functional as F
from fengshen.models.transfo_xl_paraphrase import TransfoXLTokenizer
from fengshen.models.transfo_xl_paraphrase import TransfoXLModel


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        # convert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for i in range(sorted_indices.size()[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
    return logits


def get_masks_and_position_ids(data, mem_length=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()
    # Attention mask (lower triangular).
    attention_mask = torch.ones((1, seq_length, seq_length + mem_length), device=data.device)
    attention_mask = torch.tril(torch.triu(attention_mask, 1 - seq_length + mem_length), mem_length)
    attention_mask = attention_mask.unsqueeze(1)
    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    return attention_mask, position_ids


def get_batch(context_tokens, mem_length, batch_size=1):
    tokens = context_tokens
    tokens = tokens.view(batch_size, -1).contiguous()
    # Get the masks and postition ids.
    attention_mask, position_ids = get_masks_and_position_ids(tokens, mem_length=mem_length)
    return tokens, attention_mask, position_ids


def paraphrase_generate(model,
                     tokenizer,
                     input_text,
                     device=0,
                     mem_length=512,
                     temperature=1.,
                     top_p=0.9,
                     eod_token=50000):
    ''' Generate with fixed prompt pretrained '''
    prompt = f"“{input_text}”的相似句是“"
    counter = 0
    prompt_tokens = tokenizer.encode(prompt)[:-1]
    tokens, attention_mask, position_ids = get_batch(
        torch.LongTensor(prompt_tokens), mem_length, batch_size=1)
    tokens, attention_mask, position_ids = tokens.cuda(
        device), attention_mask.cuda(device), position_ids.cuda(device)
    org_context_length = tokens.shape[-1]
    model = model.cuda(device)
    while counter < 100:
        if counter == 0:
            mems = []  # empty at the begining
            output = model(input_ids=tokens, attention_mask=attention_mask,
                           position_ids=position_ids, hidden_states=mems)
            logits, mems = output.logits, output.hidden_states
        else:
            index = org_context_length + counter
            output = model(input_ids=tokens[:, index - 1: index], position_ids=tokens.new_ones((1, 1)) * (index - 1),
                           attention_mask=tokens.new_ones(1, 1, 1, mem_length + 1, device=device,
                                                          dtype=torch.float), hidden_states=mems)
            logits, mems = output.logits, output.hidden_states
        logits = logits[:, -1]
        logits /= temperature
        logits = top_k_logits(logits, top_k=0, top_p=top_p)
        log_probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)[0]
        is_end = prev == eod_token
        if is_end:
            break
        tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
        counter += 1
    out_tokens = tokens.view(-1).contiguous().tolist()[len(prompt_tokens):]
    res = tokenizer.decode(out_tokens).split('”')[0]
    return res


if __name__ == "__main__":
    device = 1
    tokenizer = TransfoXLTokenizer.from_pretrained('IDEA-CCNL/Randeng-TransformerXL-1.B-Paraphrasing-Chinese')
    model = TransfoXLModel.from_pretrained('IDEA-CCNL/Randeng-TransformerXL-1.B-Paraphrasing-Chinese')
    input_text = "年轻教师选择农村学校，还是县城学校？"
    res = paraphrase_generate(model, tokenizer, input_text)
    print(res)

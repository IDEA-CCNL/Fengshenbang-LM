import torch
import torch.nn.functional as F
from fengshen.models.transfo_xl_paraphrase import TransfoXLModel
from fengshen.utils import top_k_logits, get_masks_and_position_ids
from transformers import T5Tokenizer


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
    device = 0
    tokenizer = T5Tokenizer.from_pretrained('IDEA-CCNL/Randeng-TransformerXL-1.1B-Paraphrasing-Chinese',
                                           eos_token='<|endoftext|>',
                                           extra_ids=0)
    model = TransfoXLModel.from_pretrained('IDEA-CCNL/Randeng-TransformerXL-1.1B-Paraphrasing-Chinese')
    input_text = "年轻教师选择农村学校，还是县城学校？"
    res = paraphrase_generate(model, tokenizer, input_text, device=device)
    print(res)

import re
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import jsonlines
from tqdm import tqdm, trange

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def filter_noise(text):
    space_pattern = '([\u4e00-\u9fa5|0-9|，|。|？|！|@|¥|……|——|《|》|“|”|、|；|：|‘|’|（|）|「|」|【|】|·|～|-|+])\s+([\u4e00-\u9fa5|0-9|，|。|？|！|@|¥|……|——|《|》|“|”|、|；|：|‘|’|（|）|「|」|【|】|·|～|-|+])'
    text = re.sub(space_pattern, r'\1\2', text)
    text = re.sub(space_pattern, r'\1\2', text)
    patterns = ['引用日期.*$', '参考资料.*$', '\[.*\]', '【.*】', '原文地址：', '原文转载：', '本文转自：', '本文摘要：', '<unk>']
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text.strip()

def get_raw_data(raw_data):
    train_data = {}
    with open(raw_data, 'r', encoding='utf8') as fh:
        for line in fh:
            line = json.loads(line)
            for key in line.keys():
                if key not in train_data.keys():
                    train_data[key] = [line[key]]
                else:
                    train_data[key].append(line[key])
    return train_data

def save_output(input_text, output, output_file):
    with jsonlines.open(output_file, mode='a') as writer:
        for text_in,text_out in zip(input_text, output):
            otc = {}
            otc['text_a'] = str(text_in)
            otc['text_b'] = str(text_out)
            writer.write(otc)

def enforce_repetition_penalty(lprobs, prev_output_tokens, repetition_penalty = 1.5):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(len(prev_output_tokens)):
        for previous_token in set(prev_output_tokens[i]):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1# batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
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
        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # logits[indices_to_remove] = filter_value
    return logits

def sample_sequence_conditional(model, length, context, latent_z=None, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0, device='cpu'):

    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for i in trange(length):
            if i == 2:
                generated[generated[:, 1] == 127, 1] = 0 
            attention_mask = model.get_attn_mask(generated.shape[1]).to(device)
            inputs = {'input_ids': generated, 'latent_state': latent_z, 'attention_mask':attention_mask, 'mems':None}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            log_probs = F.softmax(filtered_logits, dim=-1)
            if repetition_penalty != 1.0:
                enforce_repetition_penalty(log_probs, generated, repetition_penalty)
            next_token = torch.multinomial(log_probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            # pdb.set_trace()
            # if next_token[0,0].item() == decoder_tokenizer.encode('<EOS>')[0]:
            if next_token[0, 0] == 50000: # end of token 50000
                break

    return generated

def latent_code_from_text(text, tokenizer_encoder, model_vae, args, scale=1.0):
    tokenized1 = tokenizer_encoder.encode(text)
    coded = torch.Tensor([tokenized1]).long()
    with torch.no_grad():
        coded = coded.to(device)
        outputs = model_vae.encoder(coded, attention_mask=(coded > 0).float())
        pooled_hidden_fea = outputs[1]

        mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        std = logvar.mul(0.5).exp()
        eps = torch.zeros_like(std).normal_()

        return mean + torch.mul(eps, std)*scale

def text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder, prompt=None):
    bos_token = tokenizer_decoder.convert_tokens_to_ids(tokenizer_decoder.bos_token)
    context_tokens = [bos_token]

    if prompt is not None:
        context_tokens.append(tokenizer_decoder.encode(prompt)[:-1]) # remove eos token

    out = sample_sequence_conditional(
        model=model_vae.decoder,
        context=context_tokens,
        latent_z=latent_z,
        length= args.max_out_length, # Chunyuan: Fix length; or use <EOS> to complete a sentence
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device
    )

    out_tokens = out[0, :].tolist()
    out_tokens = out_tokens[1:out_tokens.index(50000)] if 50000 in out_tokens else out_tokens # remove bos and eos
    text_x1 = tokenizer_decoder.decode(out_tokens, clean_up_tokenization_spaces=True)

    return text_x1


def simulate(model_vae, tokenizer_encoder, tokenizer_decoder, args, sent_input, prompt=None):
    latent_z, _ = latent_code_from_text(sent_input, tokenizer_encoder, model_vae, args)
    text_analogy = text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder, prompt=prompt)
    
    return text_analogy

def switch(next_value, init, is_update):
    is_update = is_update.type_as(next_value)
    return (1-is_update)*init + is_update*next_value

def sample_sequence_conditional_batch(model, max_out_length, context_tokens_tensor, context_length_tensor, latent_z=None, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0, device='cpu', end_token=50000):
    org_context_length = torch.min(context_length_tensor).item()
    batch_size = context_tokens_tensor.shape[0]

    generated = context_tokens_tensor[:,:org_context_length]
    counter = org_context_length

    output_tokens_lists = []
    output_order = []
    orig_order = torch.LongTensor(list(range(batch_size)))

    with torch.no_grad():
        while counter < max_out_length:
            if counter == org_context_length+2:
                generated[generated[:,org_context_length] == 127, org_context_length] = 0 
            attention_mask = model.get_attn_mask(generated.shape[1]).to(device)
            inputs = {'input_ids': generated, 'latent_state': latent_z, 'attention_mask': attention_mask}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # if counter == org_context_length:
            #     filtered_logits[:, 43488] = -float('Inf') # forbid starting with '《'
            log_probs = F.softmax(filtered_logits, dim=-1)

            if repetition_penalty != 1.0:
                enforce_repetition_penalty(log_probs, generated, repetition_penalty)

            if any(log_probs.sum(dim=-1) <= 0.0) :
                break
            next_token = torch.multinomial(log_probs, num_samples=1).view(-1)
            next_token = switch(next_token, context_tokens_tensor[:, counter], context_length_tensor<=counter)

            if torch.all(next_token == end_token).item():
                break

            stop_idx = next_token == end_token
            output_order.extend(orig_order[stop_idx].tolist())

            finished = generated[stop_idx]
            output_tokens_lists.extend(finished.detach().cpu().tolist())
            # continue with non-ending tokens
            conti_idx = next_token != end_token
            orig_order = orig_order[conti_idx]
            generated = generated[conti_idx]
            latent_z = latent_z[conti_idx]

            next_token = next_token[conti_idx]
            context_tokens_tensor = context_tokens_tensor[conti_idx]
            context_length_tensor = context_length_tensor[conti_idx]
            batch_size = generated.shape[0]

            generated = torch.cat((generated, next_token.view(batch_size, 1)), dim=-1)
            counter += 1

        output_order.extend(orig_order.tolist())
        generated = generated.detach().cpu().tolist()
        output_tokens_lists.extend(generated)
        output_tokens_lists = [tokens[:tokens.index(end_token)] if end_token in tokens else tokens for tokens in output_tokens_lists]

        output_tokens_lists = [tokens for _,tokens in sorted(zip(output_order, output_tokens_lists))]

    return output_tokens_lists

def latent_code_from_text_batch(texts, tokenizer_encoder, model_vae, args):
    tokens_tensor_list = []
    for text in texts:
        tokens = tokenizer_encoder.encode(text)[:510]
        tokens_tensor_list.append(torch.tensor([101]+tokens+[102]))

    coded = pad_sequence(tokens_tensor_list, batch_first=True, padding_value=0).long()
    with torch.no_grad():
        coded = coded.to(device)
        pooled_hidden_fea = model_vae.encoder(coded, attention_mask=(coded > 0).float())[1]
        mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)

        std = logvar.mul(0.5).exp()
        eps = torch.zeros_like(std).normal_()

        latent_z = mean + torch.mul(eps, std)*args.std_scale

        return latent_z

def text_from_latent_code_batch(latent_z, model_vae, args, tokenizer_decoder, prompt=None):
    past = latent_z
    batch_size = latent_z.shape[0]
    bos_token = tokenizer_decoder.convert_tokens_to_ids(tokenizer_decoder.bos_token)
    end_token = tokenizer_decoder.convert_tokens_to_ids(tokenizer_decoder.eos_token)

    if prompt is not None:
        prompt = [[bos_token] + tokenizer_decoder.encode(text)[:-1] for text in prompt]
    else:
        prompt = [[bos_token]]*batch_size

    context_tokens_tensor = torch.tensor([[end_token]*args.max_out_length]*batch_size).to(device) # 2-d tensor
    context_length_tensor = torch.tensor([1]*batch_size).to(device)
    for i in range(batch_size):
        context_tokens_tensor[i,:len(prompt[i])] = torch.tensor(prompt[i]).long().to(device)
        context_length_tensor[i] = len(prompt[i])

    # length = 128 # maximum length, but not used
    out = sample_sequence_conditional_batch(
        model=model_vae.decoder,
        max_out_length= args.max_out_length, # Chunyuan: Fix length; or use <EOS> to complete a sentence
        context_tokens_tensor=context_tokens_tensor,
        context_length_tensor=context_length_tensor,
        latent_z=latent_z,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device
    )

    out_text = []
    for i, tokens in enumerate(out):
        tokens = tokens[len(prompt[i]):]
        tokens = tokens[:tokens.index(end_token)] if end_token in tokens else tokens
        text = tokenizer_decoder.decode(tokens, clean_up_tokenization_spaces=True)
        out_text.append(filter_noise(text))
    return out_text


def simulate_batch(model_vae, tokenizer_encoder, tokenizer_decoder, args, sent_inputs, prompt=None):
    latent_z = latent_code_from_text_batch(sent_inputs, tokenizer_encoder, model_vae, args)
    text_analogy = text_from_latent_code_batch(latent_z, model_vae, args, tokenizer_decoder, prompt=prompt)
    return text_analogy

def simulate_bz(model_vae, tokenizer_encoder, tokenizer_decoder, args, sent_inputs, prompt=None):
    latent_z = latent_code_from_text_batch(sent_inputs, tokenizer_encoder, model_vae, args)
    return latent_z

def my_shuffle(x, index):
    result = []
    for field in index:
        result.append(x[field])
    return result


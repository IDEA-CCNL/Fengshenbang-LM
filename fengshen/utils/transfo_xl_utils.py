# encoding=utf-8
import torch, math
import torch.nn.functional as F


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


def enforce_repetition_penalty(lprobs, prev_output_tokens, repetition_penalty=1.5):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for previous_token in set(prev_output_tokens):
        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
        if lprobs[previous_token] < 0:
            lprobs[previous_token] *= repetition_penalty
        else:
            lprobs[previous_token] /= repetition_penalty


def switch(next_value, init, is_update):  # 换成真实token
    is_update = is_update.type_as(next_value)
    return (1-is_update)*init + is_update*next_value


def get_atten_mask(batch_size, seq_length, memory_length=0):
    memory_attention_mask = torch.ones(
        (batch_size, 1, seq_length, seq_length + memory_length), dtype=torch.int16)
    memory_attention_mask = torch.tril(
        torch.triu(memory_attention_mask, 1 - seq_length + memory_length), memory_length)

    return memory_attention_mask  # [bs, 1, seq_len, seq_len+M]


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


def sample_sequence_batch(model, context_tokens_tensor, context_length_tensor, max_out_seq=None, mems=None,
                          end_token_id=None, repetition_penalty=1.0, temperature=1.0, top_k=0, top_p=0.0):
    """_summary_

    Args:
        model (_type_): _description_
        context_tokens_tensor (Tensor): [bs, seq_len]
        context_length_tensor (Tensor): [bs, ]
        max_out_seq (_type_, optional): _description_. Defaults to None.
        mems (_type_, optional): _description_. Defaults to None.
        end_token_id (_type_, optional): _description_. Defaults to None.
        repetition_penalty (float, optional): _description_. Defaults to 1.0.
        temperature (float, optional): _description_. Defaults to 1.0.
        top_k (int, optional): _description_. Defaults to 0.
        top_p (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    
    model_dtype = next(model.parameters()).dtype
    org_context_length = torch.min(context_length_tensor).item()
    batch_size = context_tokens_tensor.shape[0]
    tokens = context_tokens_tensor[:, :org_context_length]
    attention_mask = get_atten_mask(batch_size, org_context_length).cuda(context_tokens_tensor.device).to(model_dtype)
    position_ids = torch.arange(org_context_length, dtype=torch.long,
                                device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)

    counter, mem_length = 0, 0
    if mems is None:
        mems = []
    if end_token_id is None:
        end_token_id = 50000
    if max_out_seq is None:
        max_out_seq = 512

    output_tokens_lists = []
    
    # record order
    origin_order = torch.tensor(range(batch_size), device=tokens.device)
    output_order = []

    # record log_probs
    log_probs_tensor = torch.tensor([0.0] * batch_size, device=tokens.device)
    log_probs_list = []

    with torch.no_grad():
        # while counter < (max_out_seq - org_context_length):
        while counter < max_out_seq:
            index = org_context_length + counter
            if counter == 0:
                output = model.forward(input_ids=tokens, position_ids=position_ids, 
                                              attention_mask=attention_mask, hidden_states=mems)
                logits, mems = output.logits, output.hidden_states
            else:
                output = model.forward(input_ids=tokens[:, index - 1: index], position_ids=tokens.new_ones((1, 1)) * (index - 1), 
                                              attention_mask=tokens.new_ones(batch_size, 1, 1, mem_length + 1).to(model_dtype), hidden_states=mems)
                logits, mems = output.logits, output.hidden_states
            logits = logits[:, -1]
            logits /= temperature
            logits = top_k_logits(logits, top_k=top_k, top_p=top_p)
            # if repetition_penalty != 1.0:
            #     for bz in range(batch_size):
            #         enforce_repetition_penalty(logits[bz, :], tokens[bz, :], repetition_penalty)
            log_probs = F.softmax(logits, dim=-1)  # [bs, vocab_size]

            # if repetition_penalty != 1.0:
            #     for bz in range(batch_size):
            #         enforce_repetition_penalty(
            #             log_probs[bz, :], tokens[bz, :], repetition_penalty)

            prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            if index < torch.max(context_length_tensor).item():
                prev = switch(
                    prev, context_tokens_tensor[:, index], context_length_tensor <= index)
            
            for i in range(batch_size):
                if index > context_length_tensor[i] and prev[i] != end_token_id:
                    log_probs_tensor[i] += math.log(log_probs[i][prev[i]] + 1e-6) ###
                if prev[i] == end_token_id:
                    log_probs_tensor[i] /= (context_length_tensor[i].cpu() - index)

            # with torch.autocast('cpu'):
            stop_idx = prev == end_token_id
            if torch.all(stop_idx).item():
                output_order.extend(origin_order[stop_idx].tolist())
                break

            finished = tokens[stop_idx]
            output_tokens_lists.extend(finished.detach().cpu().tolist())
            log_probs_list.extend(log_probs_tensor[stop_idx].tolist())
            output_order.extend(origin_order[stop_idx].tolist())

            # continue with non-ending tokens
            conti_idx = (prev != end_token_id)
            origin_order = origin_order[conti_idx]
            tokens, prev = tokens[conti_idx], prev[conti_idx]
            context_tokens_tensor = context_tokens_tensor[conti_idx]
            context_length_tensor = context_length_tensor[conti_idx]
            log_probs_tensor = log_probs_tensor[conti_idx]
            batch_size = tokens.shape[0]
            for im in range(len(mems)):
                mems[im] = mems[im][conti_idx, :, :]

            tokens = torch.cat((tokens, prev.view(batch_size, 1)), dim=-1)

            counter += 1

    output_tokens_lists.extend(tokens.detach().cpu().tolist())
    log_probs_list.extend(log_probs_tensor.tolist())
    output_order.extend(origin_order.tolist()) ###
    output_tokens_lists = [tokens[:tokens.index(
        end_token_id)] if end_token_id in tokens else tokens for tokens in output_tokens_lists]

    output_tokens_lists = [tokens for _, tokens in sorted(zip(output_order, output_tokens_lists))]
    output_log_porbs = [prob for _, prob in sorted(zip(output_order, log_probs_list))]

    return output_tokens_lists, output_log_porbs


def sample_sequence(model, tokens, attention_mask, do_sampling=True,
                    repetition_penalty=1.0, max_out_seq=None, mems=None, end_token_id=None,
                    mem_length=0, temperature=1.0, top_k=0, top_p=0.0):
    """_summary_

    Args:
        model (_type_): _description_
        tokens (Tensor): [1, seq_len]
        attention_mask (Tensor): [1, 1, seq_len, seq_len]
        do_sampling (bool, optional): _description_. Defaults to True.
        repetition_penalty (float, optional): _description_. Defaults to 1.0.
        max_out_seq (_type_, optional): _description_. Defaults to None.
        mems (_type_, optional): _description_. Defaults to None.
        end_token (_type_, optional): _description_. Defaults to None.
        mem_length (int, optional): _description_. Defaults to 0.
        temperature (float, optional): _description_. Defaults to 1.0.
        top_k (int, optional): _description_. Defaults to 0.
        top_p (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    counter = 0
    if mems is None:
        mems = []
    if end_token_id is None:
        end_token_id = 50000
    if max_out_seq is None:
        max_out_seq = 512
    org_context_length = tokens.size(1)
    with torch.no_grad():
        # while counter < (max_out_seq - org_context_length):
        while counter < max_out_seq:
            if counter == 0:
                logits, *mems = model(input_ids=tokens, position_ids=None,
                                      attention_mask=attention_mask, mems=mems)
            else:
                index = org_context_length + counter
                logits, *mems = model(input_ids=tokens[:, index - 1: index], position_ids=None,
                                      attention_mask=tokens.new_ones(1, 1, 1, mem_length + 1), mems=mems)
            logits = logits[:, -1]
            logits /= temperature
            if do_sampling:
                logits = top_k_logits(logits, top_k=top_k, top_p=top_p)
            log_probs = F.softmax(logits, dim=-1)

            if repetition_penalty != 1.0:
                enforce_repetition_penalty(
                    log_probs[0, :], tokens[0, :], repetition_penalty)
            prev = torch.multinomial(log_probs, num_samples=1)[0]
            is_end = (prev == end_token_id)
            if is_end:
                break
            tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
            counter += 1

    output_tokens_list = tokens.detach().cpu().tolist()
    if end_token_id in output_tokens_list:
        output_tokens_list = output_tokens_list[:output_tokens_list.index(
            end_token_id)]

    return output_tokens_list[0], mems

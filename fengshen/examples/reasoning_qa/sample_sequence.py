import torch
import torch.nn.functional as F
import os

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        # convert to 1D
        #logits = logits.view(logits.size()[1]).contiguous()
        #logits = logits.contiguous()
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
        #indices_to_remove = sorted_indices[sorted_indices_to_remove]
        #logits[indices_to_remove] = filter_value
        # going back to 2D
        #logits = logits.view(1, -1).contiguous()
    return logits

def enforce_repetition_penalty(lprobs, prev_output_tokens, repetition_penalty = 1.5):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for previous_token in set(prev_output_tokens):
        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
        if lprobs[previous_token] < 0:
            lprobs[previous_token] *= repetition_penalty
        else:
            lprobs[previous_token] /= repetition_penalty

def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               transformer_xl=False,
                               mem_length=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if transformer_xl:
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = torch.tril(torch.triu(attention_mask, 1 - seq_length + mem_length), mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if not transformer_xl:
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i+1):, :(i+1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i+1):] -= (i + 1 - prev_index)
                        prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_batch(context_tokens, device, args, batch_size = 1):
    tokens = context_tokens
    tokens = tokens.view(batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)

    return tokens, attention_mask, position_ids

def switch(next_value, init, is_update):
    is_update = is_update.type_as(next_value)
    return (1-is_update)*init + is_update*next_value

def create_context(tokenizer, device, args, texts, context_func=None, max_out_seq=None, end_token=None):
    if isinstance(texts, str):
        texts = [texts]
    if context_func is None:
        context_texts = ["“"+input_text+"”的相似句是“" for input_text in texts]
    else:
        if isinstance(context_func, list):
            context_texts = []
            for func in context_func:
                context_texts.extend([func(input_text) for input_text in texts])
        else:
            context_texts = [context_func(input_text) for input_text in texts]
    if end_token is None:
        end_token = args.eod_token
    if max_out_seq is None:
        max_out_seq = args.out_seq_length

    batch_size = len(context_texts)
    context_length_tensor = torch.ones(batch_size, dtype=torch.long).to(device)
    context_tokens_tensor = torch.ones((batch_size, max_out_seq), dtype=torch.long).to(device)*end_token
    for i in range(batch_size):
        input_tokens = tokenizer.EncodeAsIds(context_texts[i]).tokenization
        if len(input_tokens) > max_out_seq:
            input_tokens = input_tokens[:max_out_seq]
        context_length_tensor[i] = len(input_tokens)
        context_tokens_tensor[i,:len(input_tokens)] = torch.LongTensor(input_tokens).to(device)
    return context_tokens_tensor, context_length_tensor

def sample_sequence_batch(model, tokenizer, context_tokens_tensor, context_length_tensor, args, device, max_out_seq=None, mems=None, end_token=None, repetition_penalty=1.5):
    org_context_length = torch.min(context_length_tensor).item()
    batch_size = context_tokens_tensor.shape[0]
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor[:,:org_context_length], device, args, batch_size)
    counter = 0
    if mems is None:
        mems = []
    if end_token is None:
        end_token = args.eod_token
    if max_out_seq is None:
        max_out_seq = args.out_seq_length

    output_tokens_lists = []
    with torch.no_grad():
        # while counter < (max_out_seq - org_context_length):
        while counter < max_out_seq:
            index = org_context_length + counter
            if counter == 0:
                logits, *mems = model(tokens, position_ids, attention_mask, *mems)
            else:
                logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((batch_size, 1)) * (index - 1),
                                      tokens.new_ones(batch_size, 1, 1, args.mem_length + 1, device=tokens.device,
                                                      dtype=torch.float), *mems)
            logits = logits[:, -1]
            logits /= args.temperature
            logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)

            if repetition_penalty != 1.0:
                for bz in range(batch_size):
                    enforce_repetition_penalty(log_probs[bz,:], tokens[bz,:], repetition_penalty)

            prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            prev = switch(prev, context_tokens_tensor[:, index], context_length_tensor<=index)

            if torch.all(prev == end_token).item():
                break

            finished = tokens[prev == end_token]
            output_tokens_lists.extend(finished.detach().cpu().tolist())
            # continue with non-ending tokens
            conti_idx = prev != end_token
            tokens = tokens[conti_idx]
            prev = prev[conti_idx]
            context_tokens_tensor = context_tokens_tensor[conti_idx]
            context_length_tensor = context_length_tensor[conti_idx]
            batch_size = tokens.shape[0]
            for im in range(len(mems)):
                mems[im] = mems[im][conti_idx, :, :]

            tokens = torch.cat((tokens, prev.view(batch_size, 1)), dim=-1)

            counter += 1

    output_tokens_lists.extend(tokens.detach().cpu().tolist())
    output_tokens_lists = [tokens[:tokens.index(end_token)] if end_token in tokens else tokens for tokens in output_tokens_lists]
    return output_tokens_lists, mems

def evalute_perplexity_batch(model, tokenizer, context_tokens_tensor, context_length_tensor, args, device, max_out_seq=None, mems=None, end_token=None):
    org_context_length = torch.min(context_length_tensor).item()
    batch_size = context_tokens_tensor.shape[0]
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor[:,:org_context_length], device, args, batch_size)
    counter = 0
    if mems is None:
        mems = []
    if end_token is None:
        end_token = args.eod_token
    if max_out_seq is None:
        max_out_seq = args.out_seq_length

    with torch.no_grad():
#         while counter < (args.out_seq_length - org_context_length):
        while counter < max_out_seq:
            index = org_context_length + counter
            if counter == 0:
                logits, *mems = model(tokens, position_ids, attention_mask, *mems)
            else:
                logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((batch_size, 1)) * (index - 1),
                                      tokens.new_ones(batch_size, 1, 1, args.mem_length + 1, device=tokens.device,
                                                      dtype=torch.float), *mems)
            logits = logits[:, -1]
            logits /= args.temperature
            logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            prev = switch(prev, context_tokens_tensor[:, index], context_length_tensor<=index)
            if torch.all(prev == end_token).item():
                break
            tokens = torch.cat((tokens, prev.view(batch_size, 1)), dim=-1)

            counter += 1

    output_tokens_lists = tokens.detach().cpu().tolist()
    output_tokens_lists = [tokens[:tokens.index(end_token)] if end_token in tokens else tokens for tokens in output_tokens_lists]
    return output_tokens_lists, mems


def sample_sequence(model, tokenizer, context_tokens_tensor, context_length, args, device, do_sampling=True, repetition_penalty=1.0, max_out_seq=None, mems=None, end_token=None):
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
    counter = 0
    if mems is None:
        mems = []
    if end_token is None:
        end_token = args.eod_token
    if max_out_seq is None:
        max_out_seq = args.out_seq_length
    org_context_length = context_length
    with torch.no_grad():
        # while counter < (max_out_seq - org_context_length):
        while counter < max_out_seq:
            if counter == 0:
                logits, *mems = model(tokens, position_ids, attention_mask, *mems)
            else:
                index = org_context_length + counter
                logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                                      tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                      dtype=torch.float), *mems)
            logits = logits[:, -1]
            logits /= args.temperature
            if do_sampling:
                logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)

            if repetition_penalty != 1.0:
                enforce_repetition_penalty(log_probs[0,:], tokens[0,:], repetition_penalty)
            prev = torch.multinomial(log_probs, num_samples=1)[0]
            is_end = prev == end_token
            if is_end:
                break
            tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
            context_length += 1
            counter += 1
            #if mpu.get_model_parallel_rank() == 0 and counter % 16 == 0:
            #    output_tokens_list = tokens.view(-1).contiguous()
            #    decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
            #    if mpu.get_model_parallel_rank() == 0 and (counter % 128 == 0 or is_end):
            #        os.system('clear')
            #        trim_decode_tokens = decode_tokens
            #        print(trim_decode_tokens, flush=True)
#     output_tokens_list = tokens.view(-1).contiguous()
    output_tokens_list = tokens.detach().cpu().tolist()
    if end_token in output_tokens_list:
        output_tokens_list = output_tokens_list[:output_tokens_list.index(end_token)] 

    return output_tokens_list[0], mems

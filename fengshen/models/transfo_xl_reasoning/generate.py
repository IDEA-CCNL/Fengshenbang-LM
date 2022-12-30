# encoding=utf-8
from typing import List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer

from fengshen.models.transfo_xl_reasoning import TransfoXLModel
from fengshen.utils import sample_sequence_batch


def en_to_zh(sentence:str):
    en_pun = u",.!?[]()<>\"\"''"
    zh_pun = u"，。！？【】（）《》“”‘’"
    table = {
        ord(f): ord(t) for f,t in zip(en_pun, zh_pun)
    }
    return sentence.translate(table)


def deduction_generate(
    model:TransfoXLModel,
    tokenizer:T5Tokenizer,
    input_text:Union[str, List[str]],
    device:int=0,
    batch_size:int=2,
    temperature:float=1.0,
    repetition_penalty:float=2.0,
    top_p:float=0.6) -> List[str]:
    """ Generate with fixed prompt of deduction """

    model = model.eval().cuda(device)
    
    if isinstance(input_text, str):
        input_text = [input_text]

    input_text = [f"<bos>{text}，因而" for text in input_text]

    input_ids = [torch.tensor(ids[:-1]) for ids in tokenizer(input_text).input_ids]
    input_length = [len(ids) for ids in input_ids]

    output = []

    for index in range(0, len(input_ids), batch_size):
        input_ids_batch = pad_sequence(
            input_ids[index: index + batch_size], batch_first=True, padding_value=50000,
        )
        input_ids_length = torch.tensor(input_length[index: index + batch_size])

        res_ids_batch, _ = sample_sequence_batch(
            model=model,
            context_tokens_tensor=input_ids_batch.cuda(device=device),
            context_length_tensor=input_ids_length.cuda(device=device),
            end_token_id=50000,
            top_k=0, top_p=top_p,
            max_out_seq=512,
            repetition_penalty=repetition_penalty,
            temperature=temperature
        )

        res_sentence = [
            en_to_zh(tokenizer.decode(ids[length:])).replace(" ", "")
            for ids, length in zip(res_ids_batch, input_length[index: index + batch_size])
        ]

        output.extend(res_sentence)

    return output


def abduction_generate(
    model:TransfoXLModel,
    tokenizer:T5Tokenizer,
    input_text:Union[str, List[str]],
    device:int=0,
    batch_size:int=2,
    temperature:float=1.0,
    repetition_penalty:float=2.0,
    top_p:float=0.6) -> List[str]:
    """ Generate with fixed prompt of abduction """

    model = model.eval().cuda(device)

    if isinstance(input_text, str):
        input_text = [input_text]

    input_text = [f"<bos>之所以{text}，是因为" for text in input_text]

    input_ids = [torch.tensor(ids[:-1]) for ids in tokenizer(input_text).input_ids]
    input_length = [len(ids) for ids in input_ids]

    output = []

    for index in range(0, len(input_ids), batch_size):
        input_ids_batch = pad_sequence(
            input_ids[index: index + batch_size], batch_first=True, padding_value=50000,
        )
        input_ids_length = torch.tensor(input_length[index: index + batch_size])

        res_ids_batch, _ = sample_sequence_batch(
            model=model,
            context_tokens_tensor=input_ids_batch.cuda(device=device),
            context_length_tensor=input_ids_length.cuda(device=device),
            end_token_id=50000,
            top_k=0, top_p=top_p,
            max_out_seq=512,
            repetition_penalty=repetition_penalty,
            temperature=temperature
        )

        res_sentence = [
            en_to_zh(tokenizer.decode(ids[length:])).replace(" ", "")
            for ids, length in zip(res_ids_batch, input_length[index: index + batch_size])
        ]

        output.extend(res_sentence)

    return output


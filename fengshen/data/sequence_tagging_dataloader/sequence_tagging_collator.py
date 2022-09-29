from dataclasses import dataclass
from torch.utils.data._utils.collate import default_collate

import copy
import torch
import numpy as np

@dataclass
class CollatorForLinear:
    args = None
    tokenizer = None
    label2id = None

    def __call__(self, samples):
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        special_tokens_count = 2
        segment_id = 0

        features=[]

        for (ex_index, example) in enumerate(samples):
            tokens = copy.deepcopy(example['text_a'])

            label_ids = [self.label2id[x] for x in example['labels']]

            if len(tokens) > self.args.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.args.max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (self.args.max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [self.label2id["O"]]
            segment_ids = [segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            label_ids = [self.label2id["O"]] + label_ids
            segment_ids = [segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_len = len(label_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [segment_id] * padding_length
            label_ids += [pad_token] * padding_length

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length
            assert len(label_ids) == self.args.max_seq_length

            features.append({
                    'input_ids':torch.tensor(input_ids),
                    'attention_mask':torch.tensor(input_mask),
                    'input_len':torch.tensor(input_len),
                    'token_type_ids':torch.tensor(segment_ids),
                    'labels':torch.tensor(label_ids),
            })

        return default_collate(features)

@dataclass
class CollatorForCrf:
    args = None
    tokenizer = None
    label2id = None

    def __call__(self, samples):
        features = []
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        special_tokens_count = 2
        segment_id = 0

        for (ex_index, example) in enumerate(samples):
            tokens = copy.deepcopy(example['text_a'])

            label_ids = [self.label2id[x] for x in example['labels']]

            if len(tokens) > self.args.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.args.max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (self.args.max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [self.label2id["O"]]
            segment_ids = [segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            label_ids = [self.label2id["O"]] + label_ids
            segment_ids = [segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_len = len(label_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [segment_id] * padding_length
            label_ids += [pad_token] * padding_length

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length
            assert len(label_ids) == self.args.max_seq_length

            features.append({
                    'input_ids':torch.tensor(input_ids),
                    'attention_mask':torch.tensor(input_mask),
                    'input_len':torch.tensor(input_len),
                    'token_type_ids':torch.tensor(segment_ids),
                    'labels':torch.tensor(label_ids),
            })
        
        return default_collate(features)


@dataclass
class CollatorForSpan:
    args = None
    tokenizer = None
    label2id = None

    def __call__(self, samples):

        features = []
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        special_tokens_count = 2
        max_entities_count = 100
        segment_id = 0

        for (ex_index, example) in enumerate(samples):
            subjects = copy.deepcopy(example['subject'])
            tokens = copy.deepcopy(example['text_a'])
            start_ids = [0] * len(tokens)
            end_ids = [0] * len(tokens)
            subject_ids = []
            for subject in subjects:
                label = subject[0]
                start = subject[1]
                end = subject[2]
                start_ids[start] = self.label2id[label]
                end_ids[end] = self.label2id[label]
                subject_ids.append([self.label2id[label], start, end])
            
            subject_ids+=[[-1,-1,-1]]*(max_entities_count-len(subject_ids))

            if len(tokens) > self.args.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.args.max_seq_length - special_tokens_count)]
                start_ids = start_ids[: (self.args.max_seq_length - special_tokens_count)]
                end_ids = end_ids[: (self.args.max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids = [segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            start_ids = [0] + start_ids
            end_ids = [0] + end_ids
            segment_ids = [segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_len = len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [segment_id] * padding_length
            start_ids += [0] * padding_length
            end_ids += [0] * padding_length

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length
            assert len(start_ids) == self.args.max_seq_length
            assert len(end_ids) == self.args.max_seq_length

            features.append({
                    'input_ids': torch.tensor(np.array(input_ids)),
                    'attention_mask': torch.tensor(np.array(input_mask)),
                    'token_type_ids': torch.tensor(np.array(segment_ids)),
                    'start_positions': torch.tensor(np.array(start_ids)),
                    'end_positions': torch.tensor(np.array(end_ids)),
                    "subjects": torch.tensor(np.array(subject_ids)),
                    'input_len': torch.tensor(np.array(input_len)),
                })
        
        return default_collate(features)


@dataclass
class CollatorForBiaffine:
    args = None
    tokenizer = None
    label2id = None

    
    def __call__(self, samples):

        features = []
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        special_tokens_count = 2
        segment_id = 0

        for (ex_index, example) in enumerate(samples):
            subjects = copy.deepcopy(example['subject'])
            tokens = copy.deepcopy(example['text_a'])

            span_labels = np.zeros((self.args.max_seq_length,self.args.max_seq_length))
            span_labels[:] = self.label2id["O"]

            for subject in subjects:
                label = subject[0]
                start = subject[1]
                end = subject[2]
                if start < self.args.max_seq_length - special_tokens_count and end < self.args.max_seq_length - special_tokens_count:
                    span_labels[start + 1, end + 1] = self.label2id[label]

            if len(tokens) > self.args.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.args.max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            span_labels[len(tokens), :] = self.label2id["O"]
            span_labels[:, len(tokens)] = self.label2id["O"]
            segment_ids = [segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            span_labels[0, :] = self.label2id["O"]
            span_labels[:, 0] = self.label2id["O"]
            segment_ids = [segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [0] * len(input_ids)
            span_mask = np.ones(span_labels.shape)
            input_len = len(input_ids)

            padding_length = self.args.max_seq_length - len(input_ids)

            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [segment_id] * padding_length
            span_labels[input_len:, :] = 0
            span_labels[:, input_len:] = 0
            span_mask[input_len:, :] = 0
            span_mask[:, input_len:] = 0
            span_mask=np.triu(span_mask,0)
            span_mask=np.tril(span_mask,10)

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length
            assert len(span_labels) == self.args.max_seq_length
            assert len(span_labels[0]) == self.args.max_seq_length

            features.append({
                    'input_ids': torch.tensor(np.array(input_ids)),
                    'attention_mask': torch.tensor(np.array(input_mask)),
                    'token_type_ids': torch.tensor(np.array(segment_ids)),
                    'span_labels': torch.tensor(np.array(span_labels)),
                    'span_mask': torch.tensor(np.array(span_mask)),
                    'input_len': torch.tensor(np.array(input_len)),
            })
        
        return default_collate(features)
import os
import time

import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
from fengshen.data.MMapIndexDataset import MMapIndexDataset


def safe_check(a, type='uint8'):
    d = {'uint8': [0, 255],
         'uint16': [0, 65535]
         }
    range = d[type]
    for l in a:
        for e in l:
            assert e >= range[0] and e <= range[1]


@dataclass
class CBartDataCollator(DataCollatorMixin):

    tokenizer: None
    return_tensors: str = "pt"

    def __init__(self, args):
        self.masked_lm = args.masked_lm
        self.encoder_loss_type = args.encoder_loss_type

    @staticmethod
    def create_decoder_inputs(encoder_inputs, encoder_labels, mask_token_id):
        """
        :param encoder_inputs: list, each element is an int
        :param encoder_labels: list, each element is an int
        :return:
        """
        decoder_inputs = []
        for i, l in zip(encoder_inputs, encoder_labels):
            if l == 0:
                decoder_inputs.append(i)
            elif l == 1:
                decoder_inputs.append(mask_token_id)
            else:
                decoder_inputs += [mask_token_id] * (l - 1)
                decoder_inputs.append(i)
        return torch.tensor(decoder_inputs, dtype=torch.long)

    @staticmethod
    def torch_call(self, features):
        encoder_inputs = [s[0] for s in features]
        encoder_labels = [s[1] for s in features]
        decoder_labels = [s[2] for s in features]

        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(
            encoder_inputs, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
        encoder_labels = pad_sequence(
            encoder_labels, batch_first=True, padding_value=-100)
        if self.encoder_loss_type == 1:  # labels for mse loss
            encoder_labels = encoder_labels.float()

        decoder_labels = pad_sequence(
            decoder_labels, batch_first=True, padding_value=-100)
        # avoid computing loss on the first token, i.e. bos_token
        decoder_labels[:, 0] = -100

        # this method is for non-autoregressive decoding.
        decoder_inputs = [self.create_decoder_inputs(
            s[0], s[1], self.tokenizer.mask_token_id) for s in features]

        # replace the eos_token_id with pad_token_id
        for i, _ in enumerate(decoder_inputs):
            decoder_inputs[i][-1] = self.tokenizer.pad_token_id

        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
        # create decoder_inputs by shifting the decoder_labels right,
        _tmp = decoder_inputs.clone()
        decoder_inputs[:, 1:] = _tmp[:, :-1]
        decoder_inputs[:, 0] = self.tokenizer.eos_token_id

        # construct labels for masked lm loss
        masked_lm_labels = decoder_labels.clone()
        masked_lm_labels[_tmp != self.tokenizer.mask_token_id] = -100

        if self.masked_lm:
            decoder_labels = masked_lm_labels

        return {
            "input_ids": encoder_inputs,
            "encoder_labels": encoder_labels,
            "decoder_input_ids": decoder_inputs,
            "labels": decoder_labels,
            "attention_mask": attention_mask,
        }


class BARTDataset(Dataset):
    def __init__(self, dataset, mode, tokenizer=None, num_labels=-1, insert_mode=-1, max_sentence_length=40,
                 encoder_loss_type=0, statistics=True):
        self.encoder_loss_type = encoder_loss_type
        assert mode in ["train", "test", 'dev']
        self.mode = mode
        if self.mode == 'test' or self.mode == 'dev':
            self.is_train = False
        else:
            self.is_train = True
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length + 2  # the bos and eos tokens
        self.input_dataset = []
        self.encoder_labels_dataset = []
        self.decoder_labels_dataset = []

        data_dict_path_format = '/cognitive_comp/gaoxinyu/data/{}/{}_synthetic_max_insert_label{}_insert_mode{}_*.pt'.format(
            dataset, mode, num_labels - 2, insert_mode)
        data_dict_paths = glob.glob(data_dict_path_format)
        for data_dict_path in data_dict_paths:
            if os.path.exists(data_dict_path):
                print(f'''Loading data from {data_dict_path}''', flush=True)
                filename = ''.join(data_dict_path.rsplit('.pt', 1))
                self.input_dataset += [MMapIndexDataset(filename + "_incorrect_input_ids_list")]
                self.encoder_labels_dataset += [MMapIndexDataset(
                    filename + "_label_ids_list")]
                self.decoder_labels_dataset += [MMapIndexDataset(
                    filename + "_target_ids_list")]
            else:
                print(
                    f'Please create the synthetic datafile {data_dict_path} with create_synthetic_data.py.')

        self.len = 0
        for ds in self.input_dataset:
            self.len += len(ds)

        # TODO make sure the encoder loss weighting logic applys to every rank !
        if statistics:
            # print('Statistics for sentence length:')
            # lengths = [len(e) for e in self.decoder_labels]
            # (unique, counts) = np.unique(lengths, return_counts=True)
            # for k, v in zip(unique,counts):
            #     print(f'sentence length{k}: {v}')
            # print('Statistics for sentence labels:')
            labels = []
            # too slow!!
            # for ds in self.encoder_labels_dataset:
            #     for i in range(0, len(ds)):
            #         labels.extend(ds.__getitem__(i))

            # use only one dataset to calc
            for i in self.encoder_labels_dataset[0]:
                labels.extend(i)
            print(len(labels))
            (unique, counts) = np.unique(labels, return_counts=True)
            all_label_counts = 0
            for k, v in zip(unique, counts):
                print(f'Label {k}: {v}')
                all_label_counts += v
            # ZZ: calculate weights for differnet labels, labels with higher numbers get lower weights proportionally!
            revert_label_weights = 1 / \
                np.array([v / all_label_counts for k, v in zip(unique, counts)])
            self.label_weights = revert_label_weights / \
                np.sum(revert_label_weights)
        else:
            # ZZ: if statistics is not triggered, manually assign weights to different class
            if num_labels == 7:
                # the cross entropy loss weighst does not need to sum to 1
                self.label_weights = [0.01, 0.05, 0.1, 0.1, 0.5, 0.5, 0.5]
            else:
                self.label_weights = [1 / num_labels] * num_labels
        print(f"label weights for encoder will be {self.label_weights}")

    def __getitem__(self, idx):
        for i in range(0, len(self.input_dataset)):
            if idx >= len(self.input_dataset[i]):
                idx -= len(self.input_dataset[i])
            else:
                break
        return torch.tensor(self.input_dataset[i].__getitem__(idx), dtype=torch.long), \
            torch.tensor(self.encoder_labels_dataset[i].__getitem__(idx), dtype=torch.long), \
            torch.tensor(self.decoder_labels_dataset[i].__getitem__(idx), dtype=torch.long)

    def __len__(self):
        return self.len

    def create_decoder_inputs(self, encoder_inputs, encoder_labels, mask_token_id):
        """
        :param encoder_inputs: list, each element is an int
        :param encoder_labels: list, each element is an int
        :return:
        """
        decoder_inputs = []
        for i, l in zip(encoder_inputs, encoder_labels):
            if l == 0:
                decoder_inputs.append(i)
            elif l == 1:
                decoder_inputs.append(mask_token_id)
            else:
                decoder_inputs += [mask_token_id] * (l - 1)
                decoder_inputs.append(i)
        return torch.tensor(decoder_inputs, dtype=torch.long)

    def create_mini_batch(self, samples):
        encoder_inputs = [s[0] for s in samples]
        encoder_labels = [s[1] for s in samples]
        decoder_labels = [s[2] for s in samples]

        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(encoder_inputs, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
        encoder_labels = pad_sequence(encoder_labels, batch_first=True, padding_value=-100)
        if self.encoder_loss_type == 1:  # labels for mse loss
            encoder_labels = encoder_labels.float()

        decoder_labels = pad_sequence(decoder_labels, batch_first=True, padding_value=-100)
        # avoid computing loss on the first token, i.e. bos_token
        decoder_labels[:, 0] = -100

        # this method is for non-autoregressive decoding.
        decoder_inputs = [self.create_decoder_inputs(
            s[0], s[1], self.tokenizer.mask_token_id) for s in samples]

        # replace the eos_token_id with pad_token_id
        for i, _ in enumerate(decoder_inputs):
            decoder_inputs[i][-1] = self.tokenizer.pad_token_id

        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
        # create decoder_inputs by shifting the decoder_labels right,
        _tmp = decoder_inputs.clone()
        decoder_inputs[:, 1:] = _tmp[:, :-1]
        decoder_inputs[:, 0] = self.tokenizer.eos_token_id

        # construct labels for masked lm loss
        masked_lm_labels = decoder_labels.clone()
        masked_lm_labels[_tmp != self.tokenizer.mask_token_id] = -100

        return {
            "input_ids": encoder_inputs,
            "encoder_labels": encoder_labels,
            "decoder_input_ids": decoder_inputs,
            "labels": decoder_labels,
            "attention_mask": attention_mask,
        }


def get_train_dev_dataset(args, tokenizer):
    trainset = BARTDataset(
        args.dataset, "train", tokenizer=tokenizer, num_labels=args.num_labels,
        insert_mode=args.insert_mode, encoder_loss_type=args.encoder_loss_type)
    testset = BARTDataset(
        args.dataset, mode='dev', tokenizer=tokenizer, num_labels=args.num_labels,
        insert_mode=args.insert_mode, encoder_loss_type=args.encoder_loss_type)
    return trainset, testset

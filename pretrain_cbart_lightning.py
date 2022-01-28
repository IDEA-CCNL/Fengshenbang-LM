import os
import torch
import argparse
from fengshen.data.mmap_datamodule import MMapDataModule
from fengshen.models.bart import CBartLightning
from fengshen.models import model_utils

from pytorch_lightning import (
    Trainer,
    loggers,
)
from torch.nn.utils.rnn import pad_sequence
from transformers.models.bert.tokenization_bert import (
    BertTokenizer
)


class CBartDataCollator():
    def __init__(self, args, tokenizer):
        self.masked_lm = args.masked_lm
        self.encoder_loss_type = args.encoder_loss_type
        self.tokenizer = tokenizer

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

    def __call__(self, samples):
        encoder_inputs = [s['incorrect_input_ids_list'] for s in samples]
        encoder_labels = [s['label_ids_list'] for s in samples]
        decoder_labels = [s['target_ids_list'] for s in samples]

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
            s['incorrect_input_ids_list'], s['label_ids_list'], self.tokenizer.mask_token_id) for s in samples]

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


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()

    args_parser = MMapDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = model_utils.add_module_args(args_parser)
    args_parser = CBartLightning.add_module_specific_args(args_parser)

    args = args_parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    collator = CBartDataCollator(args, tokenizer)
    data_module = MMapDataModule(args=args, collate_fn=collator)
    model = CBartLightning(args)

    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'), name='cbart_lightning')
    trainer = Trainer.from_argparse_args(args,
                                         logger=logger)
    trainer.fit(model, data_module)

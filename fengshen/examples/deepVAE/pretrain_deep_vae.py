import torch
import os
import random
import math
import argparse
from fengshen.data.fs_datasets.fs_datamodule import FSDataModule
from fengshen.example.deepVAE.vae_pl_module import DeepVAEModule

from pytorch_lightning import (
    Trainer,
    loggers,
)

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.nn.utils.rnn import pad_sequence


class NER_RE_Collator:
    def __init__(self, bos_token, eos_token, sep_token) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token

    def __call__(self, samples, max_len=128):
        # when len(samples) is larger than one, we need to save the sentence length info
        inputs_tensors, entity_tensors = [], []
        for sp in samples:
            # NOTE: in TD-VAE, both encoder and decoder are gpt2, thus use decoder sent twice !
            input_entities, input_ids = sp['decoder_entities'], sp['decoder_target']
            input_entities = input_entities[:max_len] + [self.sep_token]
            # shorten input_ids, based on the fact that sentence must be longer than the entities
            input_ids = [self.bos_token] + input_ids[:max_len] + [self.eos_token]
            entity_tensors.append(torch.tensor(input_entities, dtype=torch.long))
            inputs_tensors.append(torch.tensor(input_ids, dtype=torch.long))
        if not inputs_tensors or not entity_tensors:
            return None  # if all the examples in the batch exceed max_length sentence
        inputs_tensors = pad_sequence(inputs_tensors, batch_first=True, padding_value=0)
        entity_tensors = pad_sequence(entity_tensors, batch_first=True, padding_value=0)
        return inputs_tensors, entity_tensors


class TDVAECollator:
    def __init__(self, bos_token, eos_token) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, samples, max_len=120):
        # when len(samples) is larger than one, we need to save the sentence length info
        inputs = []
        for sp in samples:
            # NOTE: in TD-VAE, both encoder and decoder are gpt2, thus use decoder sent twice !
            sent_lengths, input_ids = sp['decoder_sent_lengths'], sp['decoder_target']
            potential_indices = [idx for idx, slen in enumerate(sent_lengths) if slen < max_len]
            if len(potential_indices) == 0:
                continue  # we ignore paragraphs with only one sentence split
            selected_idx = random.choice(potential_indices)
            start_pos, end_pos = sum(sent_lengths[:selected_idx]), sum(sent_lengths[:selected_idx])+sent_lengths[selected_idx]
            selected_input_ids = [self.bos_token] + input_ids[start_pos:end_pos] + [self.eos_token]
            inputs.append(torch.tensor(selected_input_ids, dtype=torch.long))
        if not inputs:
            return None  # if all the examples in the batch exceed max_length sentence
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        return inputs


class ZH_Fin_Collator:
    def __init__(self, bos_token, eos_token) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, samples, max_len=120):
        inputs = []
        for sp in samples:
            # NOTE: in TD-VAE, both encoder and decoder are gpt2, thus use decoder sent twice !
            input_ids = sp['input_ids']
            if len(input_ids) == 0:
                continue  # we ignore paragraphs with empty string
            selected_input_ids = [self.bos_token] + input_ids + [self.eos_token]
            inputs.append(torch.tensor(selected_input_ids, dtype=torch.long))
        if not inputs:
            return None
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        return inputs


class VAEModelCheckpoint:
    @ staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--monitor', default='total_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./log/', type=str)
        parser.add_argument('--filename', default='model-{epoch:2d}-{train_loss:.4f}', type=str)

        parser.add_argument('--save_top_k', default=-1, type=int)
        parser.add_argument('--every_n_train_steps', default=1000, type=float)
        parser.add_argument('--save_weights_only', default=True, type=bool)

        return parent_args

    @staticmethod
    def get_callback(args):
        return ModelCheckpoint(monitor=args.monitor,
                               save_top_k=args.save_top_k,
                               mode=args.mode,
                               every_n_train_steps=args.every_n_train_steps,
                               save_weights_only=args.save_weights_only,
                               dirpath=args.dirpath,
                               filename=args.filename)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()

    args_parser = FSDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = DeepVAEModule.add_module_specific_args(args_parser)
    args_parser = VAEModelCheckpoint.add_argparse_args(args_parser)

    args = args_parser.parse_args()
    # TODO: update this to be tokenizer specific
    # collator = NER_RE_Collator(bos_token=21128, eos_token=21129, sep_token=102)
    # collator = TDVAECollator(bos_token=21128, eos_token=21129)
    collator = ZH_Fin_Collator(bos_token=21128, eos_token=21129)

    data_module = FSDataModule(args=args, collate_fn=collator)

    train_steps = math.ceil(len(data_module.train_dataset)*args.max_epochs /
                            args.train_batchsize / args.num_nodes / args.gpus)
    model = DeepVAEModule(args, train_steps)

    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'), name='deepvae_lightning')

    save_cpt_callback = VAEModelCheckpoint.get_callback(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[save_cpt_callback, lr_monitor],
                                         logger=logger)
    trainer.fit(model, data_module)

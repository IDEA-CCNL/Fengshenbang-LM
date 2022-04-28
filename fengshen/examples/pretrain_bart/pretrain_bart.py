from transformers import BartForConditionalGeneration, T5Tokenizer, BartConfig
from pytorch_lightning import (
    LightningModule,
    Trainer,
    loggers,
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dataclasses import dataclass
import random
import os
import numpy as np
import argparse
import torch


def padding_to_maxlength(ids, max_length, pad_id):
    cur_len = len(ids)
    len_diff = max_length - len(ids)
    return ids + [pad_id] * len_diff, [1] * cur_len + [0] * len_diff


def truncate_input_sequence(document, max_num_tokens):
    total_length = len(sum(document, []))
    if total_length <= max_num_tokens:
        return document
    else:
        tokens_to_trunc = total_length - max_num_tokens
        while tokens_to_trunc > 0:
            if len(document[-1]) >= tokens_to_trunc:
                document[-1] = document[-1][:len(document[-1]) - tokens_to_trunc]
                tokens_to_trunc = 0
            else:
                tokens_to_trunc -= len(document[-1])
                document = document[:-1]
        return document


@dataclass
class TextFillingCollator:
    tokenizer: None
    max_seq_length: int = 512
    masked_lm_prob: float = 0.15

    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Bart Text Filling Collator')
        parser.add_argument('--max_seq_length', default=512, type=int)
        parser.add_argument('--masked_lm_prob', default=0.15, type=float)
        return parent_args

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.masked_lm_prob = args.masked_lm_prob

    def create_noised_input(self, tokens_x):
        masked_number = 0
        noised_x = []
        noised_sent = []
        for sent in tokens_x:
            j = 0
            noised_sent = []
            while j < len(sent):
                if random.random() < self.masked_lm_prob:
                    num_tokens_to_mask = np.random.poisson(lam=3)
                    masked_number += num_tokens_to_mask
                    if num_tokens_to_mask > 0:
                        noised_sent.append(self.tokenizer.mask_token)
                        j += num_tokens_to_mask
                    else:
                        noised_sent.append(sent[j])
                        noised_sent.append(self.tokenizer.mask_token)
                        j += 1
                else:
                    noised_sent.append(sent[j])
                    j += 1
            noised_x.append(noised_sent)
        # random.shuffle(noised_x)
        noised_x = sum(noised_x, [])

        return noised_x, masked_number

    def generate_sample(self, item):
        tokens = sum(item, [])

        x = []
        y = []

        y = tokens + [self.tokenizer.eos_token]

        # Get Masked LM predictions
        noised_tokens, _ = self.create_noised_input(item)
        # noised_tokens = token_x

        x.append(self.tokenizer.bos_token)
        x = x + noised_tokens
        x.append(self.tokenizer.eos_token)

        input_ids, attn_mask = padding_to_maxlength(
            self.tokenizer.convert_tokens_to_ids(x),
            self.max_seq_length,
            self.tokenizer.pad_token_id)
        labels, decoder_attn_mask = padding_to_maxlength(
            self.tokenizer.convert_tokens_to_ids(y), self.max_seq_length,
            self.tokenizer.pad_token_id)

        return [input_ids, labels, attn_mask, decoder_attn_mask]

    def __call__(self, samples):
        input_ids = []
        labels = []
        attn_mask = []
        decoder_attn_mask = []
        for s in samples:
            # 需要补充 bos , eos, 所以最长长度需要-2
            trunc = truncate_input_sequence(s['tokenized_text'], self.max_seq_length - 2)
            g = self.generate_sample(trunc)
            while len(g[0]) > self.max_seq_length:
                # text filling在span=0时会insert一个mask，导致input_ids超长，这个时候做二次判断再截断一次
                trunc = trunc[:-1]
                g = self.generate_sample(trunc)
            input_ids.append(g[0])
            labels.append(g[1])
            attn_mask.append(g[2])
            decoder_attn_mask.append(g[3])

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attn_mask),
            'decoder_attention_mask': torch.tensor(decoder_attn_mask),
            'labels': torch.tensor(labels),
        }


class CustomCKPT:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('ckpt call back')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./ckpt/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)
        parser.add_argument('--save_last', action='store_true', default=True)
        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=100, type=float)
        parser.add_argument('--save_weights_only', action='store_true', default=False)
        parser.add_argument('--every_n_epochs', default=1, type=int)

        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         every_n_train_steps=args.every_n_train_steps,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.dirpath,
                                         filename=args.filename,
                                         save_last=args.save_last,
                                         every_n_epochs=args.every_n_epochs)


class BartLightning(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Bart Lightning')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        return parent_parser

    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.model_config = BartConfig.from_pretrained(args.model_path)
        self.model = BartForConditionalGeneration(config=self.model_config)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batchsize * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log('train_loss', output.loss, sync_dist=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        module.model.save_pretrained(os.path.join(
            self.hparams.default_root_dir, 'hf_pretrain_model'))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    import sys
    sys.path.append('../../')
    from data.universal_datamodule import UniversalDataModule
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = BartLightning.add_module_specific_args(args_parser)
    args_parser = CustomCKPT.add_argparse_args(args_parser)
    args_parser = TextFillingCollator.add_data_specific_args(args_parser)
    args_parser.add_argument('--deepspeed')
    args_parser.add_argument('--pretrain_sp_tokenizer', type=str, )
    args = args_parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_sp_tokenizer,
                                            additional_special_tokens=['<s>', '<mask>'],
                                            extra_ids=0)
    tokenizer.bos_token = '<s>'
    tokenizer.mask_token = '<mask>'

    collator = TextFillingCollator(tokenizer, args)

    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collator)
    module = BartLightning(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'),
        name=os.path.basename(os.path.dirname(args.model_path)))
    checkpoint_callback = CustomCKPT(args).callbacks

    if args.resume_from_checkpoint is not None and \
            not os.path.exists(args.resume_from_checkpoint):
        print('--------warning no checkpoint found--------, remove args')
        del args.resume_from_checkpoint

    # autotuning
    if args.deepspeed is not None:
        os.environ['PL_DEEPSPEED_CONFIG_PATH'] = args.deepspeed

    trainer = Trainer.from_argparse_args(args, logger=logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(module, data_module)

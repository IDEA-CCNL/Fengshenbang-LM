# -*- coding: utf-8 -*-

from data_utils import (
    get_input_mask, pseudo_summary_f1, shift_tokens_right,
    padding_to_maxlength, load_stopwords)
from data.universal_datamodule import UniversalDataModule
from utils import UniversalCheckpoint
from tokenizers_pegasus import PegasusTokenizer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer, loggers, LightningModule
from transformers import PegasusForConditionalGeneration, PegasusConfig
from pegasus_utils import text_segmentate
import os
import torch
import argparse
import sys
sys.path.append('../../')


# os.environ["CUDA_VISIBLE_DEVICES"] = '6'


class FakeAbstractCollator:

    def __init__(self, tokenizer, stopwords_dict, max_enc_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_enc_length
        self.stopwords_dict = stopwords_dict

    def __call__(self, samples):
        # print("samples: ", samples)
        labels = []
        attn_mask = []
        decoder_attn_mask = []
        source_inputs = []

        for text in samples:
            texts = text["chunks"]
            text = text_segmentate(texts)
            sentence_id_vec, source, target, source_idxs, target_idxs = pseudo_summary_f1(
                text, self.stopwords_dict, self.tokenizer, self.max_seq_length,
                "rouge-l")
            source_idxs, target_idxs = get_input_mask(sentence_id_vec,
                                                      target_idxs)
            if len(source_idxs) > self.max_seq_length:
                if 2 not in source_idxs[self.max_seq_length - 1:]:
                    source_idxs = source_idxs[:self.max_seq_length]
                    source_idxs[-1] = self.tokenizer.eos_token_id
                    sys.stderr.write("Warning split long line: " + source +
                                     "\n")
                else:
                    continue

            source_idxs, attention_mask = padding_to_maxlength(
                source_idxs, self.max_seq_length, self.tokenizer.pad_token_id)
            label, target_attention_mask = padding_to_maxlength(
                target_idxs, self.max_seq_length, self.tokenizer.pad_token_id)
            # print("sample len: ", len(source_idxs))
            source_inputs.append(source_idxs)
            attn_mask.append(attention_mask)
            decoder_attn_mask.append(target_attention_mask)
            labels.append(label)
        labels = torch.tensor(labels)
        decode_input_idxs = shift_tokens_right(labels,
                                               self.tokenizer.pad_token_id,
                                               self.tokenizer.pad_token_id)
        end_token_index = torch.where(labels == self.tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx + 1:] = -100

        # print("call samples: ")
        return {
            "input_ids": torch.tensor(source_inputs),
            "attention_mask": torch.tensor(attn_mask),
            "labels": labels,
            "decoder_input_ids": decode_input_idxs,
            "decoder_attention_mask": torch.tensor(decoder_attn_mask)
        }


class PegasusChineseModel(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('Pegasus-large')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        return parent_args

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        config = PegasusConfig.from_json_file(
            os.path.join(args.model_path, "config.json"))
        print("vocab_size: ", config.vocab_size)
        self.model = PegasusForConditionalGeneration(config=config)
        print("model.num_parameters: ", self.model.num_parameters())

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader(
            )

            # Calculate total steps
            tb_size = self.hparams.train_batchsize * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * float(
                self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) //
                                tb_size) // ab_size
            print('Total training step:', self.total_steps)

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log('train_loss', output.loss, sync_dist=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1, ))
        y_true = labels.view(size=(-1, )).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank(
        ) == 0:
            self.model.save_pretrained(
                os.path.join(
                    self.trainer.checkpoint_callback.dirpath,
                    'hf_pretrained_epoch{}_step{}'.format(
                        checkpoint['epoch'], checkpoint['global_step'])))


def main():
    args_parser = argparse.ArgumentParser("Pegasus Task")

    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser = PegasusChineseModel.add_model_specific_args(args_parser)
    args_parser.add_argument('--deepspeed')
    args_parser.add_argument(
        '--stopword_path',
        default="/cognitive_comp/dongxiaoqun/project/pegasus/own/pegasus/stopwords",
        type=str)
    args_parser.add_argument('--max_seq_length', default=1024, type=int)
    args = args_parser.parse_args()

    tokenizer = PegasusTokenizer.from_pretrained(args.model_path)
    stopwords_dict = load_stopwords(args.stopword_path)
    collator = FakeAbstractCollator(tokenizer, stopwords_dict,
                                    args.max_seq_length)
    data_module = UniversalDataModule(tokenizer=tokenizer,
                                      args=args,
                                      collate_fn=collator)
    module = PegasusChineseModel(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(
        save_dir=os.path.join(args.default_root_dir, 'logs/'),
        name=os.path.basename(os.path.dirname(args.model_path)))
    checkpoint_callback = UniversalCheckpoint(args).callbacks

    # autotuning
    if args.deepspeed is not None:
        os.environ['PL_DEEPSPEED_CONFIG_PATH'] = args.deepspeed

    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(module, data_module)


if __name__ == '__main__':
    main()

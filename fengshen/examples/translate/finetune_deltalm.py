#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../../')

from typing import List
from mosestokenizer import MosesDetokenizer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer, loggers, LightningModule
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils import UniversalCheckpoint
from fengshen.models.deltalm.modeling_deltalm import DeltalmForConditionalGeneration
from fengshen.models.deltalm.tokenizer_deltalm import DeltalmTokenizer
from fengshen.models.model_utils import add_module_args, add_inverse_square_args
from sacrebleu.metrics import BLEU
from pytorch_lightning.utilities import rank_zero_info
import logging
import os
import torch
import argparse
import json
import pandas as pd


# import sentencepiecos.environ["CUDA_VISIBLE_DEVICES"] = '5'e
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


logger = logging.getLogger(__name__)


EVAL_BLEU_ORDER = 4


def calc_bleu_from_stats(sentence_stats: pd.DataFrame) -> BLEU:
    corpus_stats = sentence_stats.sum(axis=0)
    smooth = {"smooth_method": "exp"}
    corpus_bleu = BLEU.compute_bleu(
        correct=[
            corpus_stats.correct_1_grams,
            corpus_stats.correct_2_grams,
            corpus_stats.correct_3_grams,
            corpus_stats.correct_4_grams,
        ],
        total=[
            corpus_stats.total_1_grams,
            corpus_stats.total_2_grams,
            corpus_stats.total_3_grams,
            corpus_stats.total_4_grams,
        ],
        sys_len=corpus_stats.translation_length,
        ref_len=corpus_stats.reference_length,
        **smooth
    )
    return corpus_bleu


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    # logger.debug("before target.dim() == lprobs.dim(): ", target.dim(), lprobs.dim())
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    # logger.debug("After target.dim() == lprobs.dim(): ", target.dim(), lprobs.dim())
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    # if -100 in taget
    # logger.debug("nll_loss: ", nll_loss)
    # logger.debug("smooth_loss: ", smooth_loss)
    # logger.debug("************** nll_loss ***************")
    # logger.debug(nll_loss)
    # logger.debug("************** smooth_loss ***************")
    # logger.debug(smooth_loss)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        # logger.debug("******************** pad_mask **********************")
        # logger.debug(pad_mask)
        # logger.debug("pad_mask.size: ", pad_mask.size())
        logger.debug(f"ori_nll_loss: {nll_loss.sum()}")
        logger.debug(f"ori_smooth_loss: {smooth_loss.sum()}")
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    logger.debug("nll_loss.sum: %f", nll_loss)
    logger.debug("smooth_loss.sum: %f", smooth_loss)
    eps_i = epsilon / (lprobs.size(-1) - 1)
    # logger.debug("epsilon: ", epsilon)
    # logger.debug("eps_i: ", eps_i)
    # logger.debug("lprobs.size(-1): ",lprobs.size(-1))
    logger.debug("ignore_index: %d", ignore_index)
    # logger.debug(target)
    logger.debug("target.size: %s", target.size())
    valid_length = target.ne(ignore_index).sum()
    unvalid_length = target.eq(ignore_index).sum()
    logger.debug("%d, %d", valid_length.item(), unvalid_length.item())
    loss = ((1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss) / valid_length.item()
    # logger.debug("loss: ", loss)

    return loss, nll_loss


class DataCollator:
    def __init__(self, model, tokenizer, max_enc_length, max_dec_length):
        self.tokenizer = tokenizer
        self.max_enc_length = max_enc_length
        self.max_dec_length = max_dec_length
        self.model = model

    def __call__(self, batch_samples):
        # logger.debug("samples: ", batch_samples)
        # print("samples: ", batch_samples)
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            if len(sample["src"]) != 0:
                batch_inputs.append(sample["src"])
                batch_targets.append(sample["tgt"])
        batch_data = self.tokenizer(
            batch_inputs,
            padding='max_length',
            max_length=self.max_enc_length,
            truncation=True,
            return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch_targets,
                padding='max_length',
                max_length=self.max_dec_length,
                truncation=False,
                return_tensors="pt"
            )["input_ids"]
            # batch_data['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(labels)
            end_token_index = torch.where(labels == self.tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx+1:] = -100
            batch_data['labels'] = labels

        batch_data['src'] = [sample['src'] for sample in batch_samples]
        batch_data['tgt'] = [sample['tgt'] for sample in batch_samples]

        # logger.debug(batch_data)
        return batch_data


class FinetuneTranslation(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('deltalm-base finetune')
        parser.add_argument('--label_smoothing', default=0.1, type=float)
        return parent_args

    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = DeltalmForConditionalGeneration.from_pretrained(args.model_path)
        logger.debug("model.num_parameters: %d", self.model.num_parameters())
        self.tokenizer = tokenizer
        assert self.tokenizer, "tokenizer is None!"
        self.blue_metric = BLEU()
        self.sufficient_stats: List[List[int]] = []
        self.label_smoothing = self.args.label_smoothing
        self.mose_decode = MosesDetokenizer()

        if self.args.label_smoothing != 0:
            self.loss_fn = label_smoothed_nll_loss

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
            logger.debug('Total training step: %d', self.total_steps)

    def configure_optimizers(self):
        # if self.args.use_default_configure:
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)

    def training_step(self, batch, batch_idx):
        if self.label_smoothing == 0:
            output = self.model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
            # logger.debug("******************* output *******************")
            # logger.debug(output)
            # logger.debug("batch_logits size: ", output["logits"].size())
            # logger.debug("batch_label size: ", batch['labels'].size())

            self.log('train_loss', output.loss, sync_dist=True)
            return output.loss

        # TODO label_smoothing should be implemented at here
        else:
            # logger.debug("label_smoothing: ", self.label_smoothing)
            logger.debug("******************* before poped labels **************")
            logger.debug(batch["labels"])
            logger.debug("before_poped_labe size: %s", batch["labels"].size())
            # labels = batch.pop('labels')
            labels = batch["labels"]
            # logger.debug("******* labels **********")
            # logger.debug("labels.size: ", bels.size())
            output = self.model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch["labels"],
                                use_cache=False)

            logger.debug("output is dict: %d", isinstance(output, dict))
            logits = output["logits"]
            logger.debug("******************* output *******************")
            logger.debug(output)
            logger.debug("batch_logits size: %s", output["logits"].size())
            logger.debug("batch_label size: %s", labels.size())
            logger.debug(labels)
            # logger.debug("logits.size: ", logits.size())
            m = torch.nn.LogSoftmax(dim=-1)
            lprobs = m(logits.float())
            loss, _ = self.loss_fn(lprobs.view(-1, lprobs.size(-1)), labels.view(-1),
                                   self.label_smoothing, self.tokenizer.pad_token_id)
            self.log('train_loss', loss, sync_dist=True)
            return loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1, ))
        y_true = labels.view(size=(-1, )).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / labels.size()[0]
        return acc

    def get_sufficient_stats(self, translations: List[str], references: List[str]) -> pd.DataFrame:
        assert len(translations) == len(references), (
            f"There are {len(translations)} translated sentences "
            f"but {len(references)} reference sentences"
        )

        # for sentence, ref in zip(translations, references):

        sentence_bleu = self.blue_metric.corpus_score(translations, [references])
        self.sufficient_stats.append(
            [
                # Number of correct 1-grams, .., 4-grams
                sentence_bleu.counts[0],
                sentence_bleu.counts[1],
                sentence_bleu.counts[2],
                sentence_bleu.counts[3],
                # Total number of 1-grams, .., 4-grams
                sentence_bleu.totals[0],
                sentence_bleu.totals[1],
                sentence_bleu.totals[2],
                sentence_bleu.totals[3],
                # Length of translated sentence.
                sentence_bleu.sys_len,
                # Length of reference sentence.
                sentence_bleu.ref_len,
            ]
        )

    def on_validation_start(self) -> None:
        # rm file at validation start
        prefix, ext = os.path.splitext(self.hparams.output_save_path)
        file_path_rank = '{}_{}{}'.format(
            prefix,
            self.trainer._accelerator_connector.cluster_environment.
            global_rank(), ext)
        if os.path.exists(file_path_rank):
            logger.debug('rm {}'.format(file_path_rank))
            os.remove(file_path_rank)

    def validation_step(self, batch, batch_idx):

        def postprocess_text(preds, labels):
            # preds = [pred.strip() for pred in preds]
            preds = list(map(lambda x: self.mose_decode(x.strip().split()), preds))
            labels = list(map(lambda x: self.mose_decode(x.strip().split()), labels))
            # labels = [[label.strip()] for label in labels]

            return preds, labels

        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.max_dec_length)

        preds = self.tokenizer.batch_decode(generated_ids,
                                            skip_special_tokens=True)
        labels = torch.where(batch['labels'] != -100, batch['labels'],
                             self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(labels,
                                             skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(preds, labels)

        # save preds for every rank
        prefix, ext = os.path.splitext(self.hparams.output_save_path)
        file_path_rank = '{}_{}{}'.format(
            prefix,
            self.trainer._accelerator_connector.cluster_environment.
            global_rank(), ext)
        self.save_prediction_to_file(preds=decoded_preds,
                                     sources=batch['src'],
                                     targets=decoded_labels,
                                     file_path=file_path_rank)

        self.get_sufficient_stats(decoded_preds, decoded_labels)

        # batch_bleu = self.blue_metric.corpus_score(decoded_preds, [decoded_labels]).score
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)
        # self.log('valid_batch_bleu', round(batch_bleu, 2), sync_dist=True)

    def validation_epoch_end(self, outputs):
        rank_zero_info("***** Validation results *****")
        sentence_states = pd.DataFrame(
            self.sufficient_stats,
            columns=[
                "correct_1_grams",
                "correct_2_grams",
                "correct_3_grams",
                "correct_4_grams",
                "total_1_grams",
                "total_2_grams",
                "total_3_grams",
                "total_4_grams",
                "translation_length",
                "reference_length",
            ]
        )

        computed_bleu = calc_bleu_from_stats(sentence_states)
        # logger.debug(self.sufficient_stats)
        logger.debug("computed_bleu: %s", computed_bleu)
        rank_zero_info("valid_sacrebleu_epoch_computed= {}\n".format(computed_bleu.score))
        self.log('valid_sacrebleu_epoch_computed', computed_bleu.score, sync_dist=True)
        self.sufficient_stats = []

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank(
        ) == 0:
            self.model.save_pretrained(
                os.path.join(
                    self.trainer.checkpoint_callback.dirpath,
                    'finetuned_epoch{}_step{}'.format(
                        checkpoint['epoch'], checkpoint['global_step'])))

    def save_prediction_to_file(self, preds, sources, targets, file_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            for idx, pred in enumerate(preds):
                source = sources[idx]
                target = targets[idx]
                tmp_result = dict()
                tmp_result['pred'] = pred
                tmp_result['source'] = source
                tmp_result['label'] = target
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data + '\n')

    def predict_step(self, batch, batch_idx):
        # print(batch)
        texts = batch['src']
        # output summary and metrics
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.max_dec_length
        )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        labels = torch.where(batch['labels'] != -100, batch['labels'],
                             self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        self.save_prediction_to_file(preds, texts, labels, self.hparams.output_save_path)


def configure_logger(logging_lever=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging_lever)


def main():
    args_parser = argparse.ArgumentParser("Pegasus Task")
    args_parser.add_argument('--do_eval_only',
                             action='store_true',
                             default=False)
    args_parser.add_argument('--early_stopping_callback',
                             action='store_true',
                             default=False)
    args_parser.add_argument('--pretrained_model_path',
                             default='facebook/mbart',
                             type=str)
    args_parser.add_argument('--output_save_path',
                             default='predict.json',
                             type=str)
    args_parser.add_argument('--max_enc_length', default=512, type=int)
    args_parser.add_argument('--max_dec_length', default=512, type=int)

    # * Args for data preprocessing
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)

    # * Args for training
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser = FinetuneTranslation.add_model_specific_args(args_parser)
    args_parser = add_module_args(args_parser)
    args_parser = add_inverse_square_args(args_parser)

    args = args_parser.parse_args()

    tokenizer = DeltalmTokenizer.from_pretrained(args.model_path)
    model = FinetuneTranslation(args, tokenizer)
    collator = DataCollator(model.model, tokenizer, args.max_enc_length, args.max_dec_length)
    data_model = UniversalDataModule(tokenizer=tokenizer,
                                     args=args,
                                     collate_fn=collator)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    configure_logger(logging_lever=logging.INFO)

    if not args.do_eval_only:

        lr_monitor = LearningRateMonitor(logging_interval='step')
        tensorboard_logger = loggers.TensorBoardLogger(
            save_dir=os.path.join(args.default_root_dir, 'logs/'),
            name=os.path.basename(os.path.dirname(args.model_path)))
        checkpoint_callback = UniversalCheckpoint(args).callbacks
        trainer = Trainer.from_argparse_args(
            args, logger=tensorboard_logger, callbacks=[lr_monitor, checkpoint_callback])
        trainer.fit(model, data_model)

    else:
        trainer = Trainer.from_argparse_args(args)
        # trainer.validate(model, data_model)
        trainer.predict(model, data_model)


if __name__ == '__main__':
    main()

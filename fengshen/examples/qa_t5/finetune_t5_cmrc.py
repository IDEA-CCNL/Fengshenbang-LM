# -*- encoding: utf-8 -*-
'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   finetune_t5_cmrc.py
@Time    :   2022/10/28 19:57
@Author  :   He Junqing
@Version :   1.0
@Contact :   hejunqing@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''
# here put the import lib

import pytorch_lightning as pl
import os
import sys
import time
import torch
import argparse
from collections import Counter
from fengshen.utils.utils import chinese_char_tokenize
from fengshen.data.universal_datamodule import UniversalDataModule
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import MT5ForConditionalGeneration, T5Tokenizer, MT5Config
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import corpus_bleu

torch.cuda.empty_cache()


class QAFinetuneModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group("BaseModel")
        parser.add_argument("--prediction_res_path", default=None, type=str)
        parser.add_argument(
            "--decode_strategy",
            default="greedy",
            choices=["beamsearch", "sampling", "greedy"],
        )
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.formator = args.formator
        self.max_target_length = args.max_target_length
        self.decode_strategy = args.decode_strategy
        self.rouge_metric = ROUGEScore(
            rouge_keys=("rougeL", "rouge1", "rouge2"), normalizer=lambda x: x
        )
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

        self.model = MT5ForConditionalGeneration.from_pretrained(
            args.pretrained_model_path
        )
        print("using MT5 model")

        if args.tokenizer_type == "t5_tokenizer":
            self.tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path)
            print("vocab_size:", len(self.tokenizer))
            # self.tokenizer.add_special_tokens(special_token_dict)
            # print('add special tokens to tokenizer,vocab size:',len(self.tokenizer))
        else:
            print("now only the t5_tokenizer is supported")
        self.bleu_val = []

    def setup(self, stage=None) -> None:

        if stage == "fit":
            train_loader = (
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches * float(
                    self.trainer.max_epochs
                )
                self.total_steps = (
                    len(train_loader.dataset) * self.trainer.max_epochs // tb_size
                ) // ab_size
            else:
                self.total_steps = (
                    self.trainer.max_steps // self.trainer.accumulate_grad_batches
                )

            print("Total steps: {}".format(self.total_steps))
        # return super().setup(stage)

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers

        return configure_optimizers(self)

    def on_save_checkpoint(self, checkpoint) -> None:
        # Save the current loop info in the mid of epoch
        # if you lightning <= 1.6.0  uncomment the line below
        # checkpoint['loops'] = self.trainer.checkpoint_connector._get_loops_state_dict()
        if (
            self.trainer.global_rank == 0
            and self.trainer.global_step % self.hparams.every_n_train_steps == 0
        ):
            self.model.save_pretrained(
                os.path.join(
                    self.trainer.checkpoint_callback.dirpath,
                    "hf_pretrained_epoch{}_step{}".format(
                        self.trainer.current_epoch, self.trainer.global_step
                    ),
                )
            )

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if "global_samples" in checkpoint:
            self.consumed_samples = checkpoint["global_samples"]
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset

    def training_step(self, batch, batch_idx):  # todo: change
        if self.formator == "t5style":
            output = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                decoder_input_ids=batch["decoder_input_ids"],
            )
        else:
            output = self.model(
                input_ids=batch["input_ids"],
                input_token_type=batch["token_types"],
                labels=batch["labels"],
                decoder_input_ids=batch["decoder_input_ids"],
            )
        # print(output.logits)
        acc = self.comput_metrix(output.logits, batch["labels"])
        grad = get_gradient_norm(self.model)
        self.log("train_loss", output.loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)
        self.log("train_grad", grad, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        pred_ids = self.model.generate(
            input_ids=batch["input_ids"], max_new_tokens=self.max_target_length
        )

        acc = self.comput_metrix(output.logits, batch["labels"])
        # print(output.logits.shape)
        self.log("val_loss", output.loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)
        batch_labels = torch.where(
            batch["labels"] != -100, batch["labels"], self.tokenizer.pad_token_id
        )

        ppl = torch.exp(output.loss)
        self.log("val_ppl", ppl, sync_dist=True)
        pred_tokens = self.tokenizer.batch_decode(
            pred_ids, cleanup_tokenization_space=True, skip_special_tokens=True
        )
        label_tokens = self.tokenizer.batch_decode(
            batch_labels, cleanup_tokenization_space=True, skip_special_tokens=True
        )
        pred_sentences = list(map(remove_pad, pred_tokens))
        # print(label_tokens)
        self.bleu_val.append(compute_bleu(pred_sentences, [[t] for t in label_tokens]))
        candidate = [
            chinese_char_tokenize(p).lstrip("<extra_id_0>") for p in pred_tokens
        ]
        target = [
            generate_sentence(chinese_char_tokenize(sent)).lstrip("<extra_id_0>")
            for sent in label_tokens
        ]
        self.rouge_metric.update(preds=candidate, target=target)
        f1 = compute_f1(candidate, label_tokens)
        self.log("val_f1", f1, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        n = len(self.bleu_val)
        avg_bleu = float(sum(self.bleu_val)) / n
        print("bleu:", avg_bleu)
        self.log("val_bleu", avg_bleu)
        self.bleu_val = []
        rouge_dict = self.rouge_metric.compute()
        # reset the metric after once validation
        self.rouge_metric.reset()
        for k, v in rouge_dict.items():
            self.log("val_{}".format(k), v, sync_dist=True)
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            print("rouge:\n", rouge_dict)
        return

    def predict_step(self, batch, batch_idx):
        num_beams = 1
        do_sample = False
        top_p = None
        if self.decode_strategy == "beamsearch":
            num_beams = 10
        elif self.decode_strategy == "sampling":
            num_beams = 4
            top_p = 0.9
            do_sample = True

        prediction_dic = self.model.generate(
            input_ids=batch["input_ids"],
            max_new_tokens=self.max_target_length,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p,
            no_repeat_ngram_size=3,
            return_dict_in_generate=True,
            output_scores=True,
        )
        output = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        prediction_ids = prediction_dic["sequences"]
        loss_tensor = self.loss_func(output.logits.transpose(1, 2), batch["labels"])
        indexes = torch.where(batch["labels"] == self.tokenizer.eos_token_id)[1]
        loss = torch.sum(loss_tensor, dim=1) / indexes
        return {
            "input_ids": batch["input_ids"],
            "predict_ids": prediction_ids,
            "labels": batch["labels"],
            "decoder_inputs": batch["decoder_input_ids"],
            "loss": loss,
        }

    def save_preditions(self, result, args):
        with open(args.prediction_res_path, "w", encoding="utf8") as fw:
            preditions = []
            labels = []
            for batch in result:
                print(batch.keys())
                batch_labels = torch.where(
                    batch["labels"] != -100,
                    batch["labels"],
                    self.tokenizer.pad_token_id,
                )
                for i in range(len(batch["input_ids"])):
                    context = self.tokenizer.decode(
                        batch["input_ids"][i],
                        skip_special_tokens=True,
                        cleanup_tokenization_space=True,
                    )
                    pred = self.tokenizer.decode(
                        batch["predict_ids"][i],
                        cleanup_tokenization_space=True,
                        skip_special_tokens=True,
                    )
                    target = generate_sentence(
                        self.tokenizer.batch_decode(
                            batch_labels[i], cleanup_tokenization_space=True
                        )
                    )
                    pred = pred.lstrip("<extra_id_0>")
                    target = target.lstrip("<extra_id_0>")
                    self.rouge_metric.update(
                        preds=chinese_char_tokenize(pred),
                        target=chinese_char_tokenize(target),
                    )
                    preditions.append(list(pred))
                    labels.append([list(target)])
                    fw.write("context:" + "".join(context) + "\n")
                    fw.write("pred:" + pred + "\n")
                    fw.write("target" + target + "\n")
                    fw.write("loss:{:.6f}\n".format(batch["loss"][i].item()))
                    fw.write("\n")
            bleu = compute_bleu(preditions, labels)
            fw.write("bleu:{}".format(bleu))
        print("finish prediction, saved in {}".format(args.prediction_res_path))
        return preditions, labels

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_true = labels.float()
        pad_num = torch.sum(torch.eq(labels, -100))
        corr = torch.eq(y_pred, y_true)
        acc = (torch.sum(corr.float()) - pad_num) / (
            y_true.view(size=(-1,)).shape[0] - pad_num
        )
        return acc


class PredictDataModule(UniversalDataModule):

    def predict_dataloader(self):
        return self.test_dataloader()


def main():

    total_parser = argparse.ArgumentParser("Finetune Dialogue model.")
    total_parser.add_argument("--do_eval_only", action="store_true", default=False)
    total_parser.add_argument("--pretrained_model_path", default=None, type=str)
    total_parser.add_argument("--new_vocab_path", default=None, type=str)
    total_parser.add_argument(
        "--tokenizer_type",
        default="t5_tokenizer",
        choices=["t5_tokenizer", "bert_tokenizer"],
    )
    total_parser.add_argument("--train_split_size", default=0.995, type=int)
    total_parser.add_argument("--preprocessing_num_workers", default="10", type=int)
    total_parser.add_argument("--ckpt_path", default=None, type=str)
    total_parser.add_argument("--use_cache", default=False, type=bool)
    total_parser.add_argument(
        "--formator", default="dialog", choices=["dialog", "ccqa", "t5style"]
    )

    sys.path.append("../../../")

    from fengshen.utils.universal_checkpoint import UniversalCheckpoint
    from qa_dataset import T5StyleDataset, TextGenCollator

    total_parser = T5StyleDataset.add_data_specific_args(total_parser)
    total_parser = UniversalDataModule.add_data_specific_args(
        total_parser
    )  # TaskDataModel
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
    total_parser = QAFinetuneModel.add_model_specific_args(
        total_parser
    )  # todo: check names

    args = total_parser.parse_args()
    print("Argument parse success.")
    print("superviseT5DataModel load start {}".format(get_time_str()))

    config = MT5Config.from_pretrained(args.pretrained_model_path)
    collate_fn = TextGenCollator(
        config=config,
        pad_token_id=config.pad_token_id,
        decoder_start_token_id=config.decoder_start_token_id,
        formator=args.formator)
    if not args.do_eval_only:
        datasets = {'train': T5StyleDataset(args.train_file, args, load_data_type=0, data="train"),
                    'validation': T5StyleDataset(args.val_file, args, load_data_type=0, data="dev")}

        model = QAFinetuneModel(args)
        print("superviseT5DataModel load end {}".format(get_time_str()))

        data_model = UniversalDataModule(
            tokenizer=None, args=args, collate_fn=collate_fn, datasets=datasets
        )
        print('data loaded')
        checkpoint_callback = UniversalCheckpoint(args)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        logger = loggers.TensorBoardLogger(
            save_dir=os.path.join(args.default_root_dir, "logs/")  # TOCHANGE
        )
        trainer = Trainer.from_argparse_args(
            args, logger=logger, callbacks=[checkpoint_callback, lr_monitor]
        )
        trainer.fit(model, data_model)
    else:
        datasets = {'test': T5StyleDataset(args.test_file, args, load_data_type=0, data="test")}

        data_model = PredictDataModule(
            tokenizer=None, args=args, collate_fn=collate_fn, datasets=datasets
        )

        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path)
        model = QAFinetuneModel(args=args)
        trainer = Trainer.from_argparse_args(args)
        result = trainer.predict(model, data_model, ckpt_path=args.ckpt_path)
        predictions, labels = model.save_preditions(result, args)
        sample = result[0]  # first_batch
        batch_labels = torch.where(
            sample["labels"] != -100, sample["labels"], model.tokenizer.pad_token_id
        )
        for i in range(4):
            print(tokenizer.batch_decode(sample["input_ids"][i]))
            print(tokenizer.batch_decode(sample["predict_ids"][i]))
            print(tokenizer.batch_decode(batch_labels[i]))


def compute_f1(cand, ref):
    f1_score = []
    for p, t in zip(cand, ref):
        p_tokens = p.split()
        t_tokens = t.split()
        common = Counter() & Counter(t.split())
        num_same = sum(common.values())
        if len(t_tokens) == 0 or len(p_tokens) == 0:
            f1 = int(p == t)
        elif num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(p_tokens)
            recall = 1.0 * num_same / len(t_tokens)
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)
            f1_score.append(f1)
        f1 = sum(f1_score) / float(len(cand))
        return f1


def generate_sentence(raw_list):
    words = []
    i = 0
    while i < len(raw_list) and raw_list[i] != "</s>":
        words.append(raw_list[i])
        i += 1
    return "".join(words)


def remove_pad(raw_text, ref=False):
    if ref:
        return [raw_text.lstrip("<pad>")]
    else:
        return raw_text.lstrip("<pad>")


def compute_bleu(preditions, labels):

    score_nltk = corpus_bleu(labels, preditions)
    return score_nltk


def get_gradient_norm(model):
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


if __name__ == "__main__":
    main()

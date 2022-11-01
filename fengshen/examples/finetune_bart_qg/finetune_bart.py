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
@File    :   finetune_bart.py
@Time    :   2022/10/28 18:23
@Author  :   Qi Yang
@Version :   1.0
@Contact :   yangqi@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''


from fengshen.models.model_utils import configure_optimizers
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.utils import chinese_char_tokenize
from utils import truncate_sequence, white_space_fix
from utils import LabelSmoothingCrossEntropy
import sys
import os
import torch
import argparse
import pytorch_lightning as pl
from dataclasses import dataclass
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import BartForConditionalGeneration
from transformers import BertTokenizer, AutoTokenizer
from torchmetrics.text.rouge import ROUGEScore
sys.path.append('../../../')


@dataclass
class QGT5Collator:
    @ staticmethod
    def add_data_specific_args(parent_args):
        # the hyperparameters should be determined according to the max length of context in dataset
        parser = parent_args.add_argument_group('BART DIalo Collator')
        parser.add_argument('--max_seq_length', default=512, type=int)
        parser.add_argument('--max_src_length', default=32, type=int)
        parser.add_argument('--max_kno_length', default=416, type=int)
        parser.add_argument('--max_tgt_length', default=64, type=int)
        parser.add_argument('--mask_ans_style',
                            default='normal',
                            type=str,
                            choices=['normal', 'unmask', 'anstoken', 'postag', 'anstoken_multispan', 'postag_multispan', 'normal_multispan'])
        return parent_args

    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.print_example = True
        self.mask_ans_style = args.mask_ans_style
        self.do_eval_only = args.do_eval_only
        self.tokenizer_type = args.tokenizer_type

    def encode(self, x, y):
        if self.tokenizer_type == "bert":
            x = x
            y = y
        else:
            # t5 sentence piece
            x = self.tokenizer.bos_token + x + self.tokenizer.eos_token
            y = y + self.tokenizer.eos_token

        encoder_input = self.tokenizer.encode_plus(
            x,
            max_length=self.args.max_kno_length + self.args.max_src_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        decoder_output = self.tokenizer.encode_plus(
            y,
            max_length=self.args.max_tgt_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        return encoder_input, decoder_output

    def mask(self, s):
        def replace_span(source, target, sptoken):
            ans_bos, ans_eos = s["ans_span"][0]
            return source[:ans_bos] + sptoken + source[ans_eos:]

        def replace_all(source, target, sptoken):
            return source.replace(target, sptoken)

        if 'multispan' in self.mask_ans_style:
            fn = replace_all
        else:
            fn = replace_span

        # unmask: 北京是中国的首都
        if 'unmask' in self.mask_ans_style:
            return s["context"]

        # normal: 北京是 <mask> 的首都
        if 'normal' in self.mask_ans_style:
            self.anstoken = self.tokenizer.mask_token
            masked_context = fn(s["context"], s["answer"][0], self.anstoken)
            return masked_context

        # anstoken: 北京是 [ANS] 的首都
        if 'anstoken' in self.mask_ans_style:
            anstoken_dict = {
                "bert": "[ANS]",
                "bart": "<ans>"
            }
            self.anstoken = anstoken_dict[self.tokenizer_type]
            masked_context = fn(s["context"], s["answer"][0], self.anstoken)
            return masked_context

        # postag: 北京是 <beg> 中国 <eos> 的首都
        if 'postag' in self.mask_ans_style:
            begtoken, endtoken = "<beg>", "<eos>"
            self.anstoken = begtoken + s["answer"][0] + endtoken
            masked_context = fn(s["context"], s["answer"][0], self.anstoken)
            return masked_context

        return masked_context

    def prompt(self, context, answer, question):
        pre_prompt, mid_prompt, post_prompt = "知识:", "回答:", "问题:"  # prompt

        context = truncate_sequence(context, self.args.max_kno_length-len(pre_prompt)-1)

        # used in squad-2.0
        # noted that src and tgt is reversed in qg
        answer = truncate_sequence(answer, self.args.max_src_length - len(mid_prompt)-1)
        question = truncate_sequence(question, self.args.max_tgt_length-len(post_prompt)-1)

        x_trunc = f'{pre_prompt}{context}{mid_prompt}{answer}'
        y_trunc = f'{post_prompt}{question}'
        return x_trunc, y_trunc

    def __call__(self, samples):
        """
        ans_num = 1 适用于 Train 数据只有 1 条 answer 取第一条情况
        ans_num > 1 适用于 Dev 数据有多条 answer 情况
        Input:
        input_ids: input_ids (text + answer)
        attn_mask: input attn mask
        labels:   decoder_ids (question)
        """
        input_ids, attn_mask, labels = [], [], []
        ans, qes, ctx, ans_spans, idxs, imp = [], [], [], [], [], []

        for s in samples:
            if self.do_eval_only:
                # log origin answer to compare
                ans.append(s["answer"])
                qes.append(s["question"])
                ctx.append(s["context"])
                ans_spans.append(s["ans_span"])
                idxs.append(s["idx"])

            if "is_impossible" in s:
                imp.append(s["is_impossible"])
            else:
                imp.append(False)  # SQUAD 1.0 don't have is_impossible

            if not s["is_impossible"]:  # have ans and ans_span
                context = self.mask(s)
                answer = s["answer"][0]
                question = s["question"]
            else:  # no ans and ans_span
                context = s["context"]
                answer = "无答案"
                question = s["question"]

            x_trunc, y_trunc = self.prompt(context, answer, question)
            encoder_input, decoder_output = self.encode(x_trunc, y_trunc)

            input_ids.append(encoder_input["input_ids"])
            attn_mask.append(encoder_input["attention_mask"])
            labels.append(decoder_output["input_ids"])

        labels = torch.cat(labels)
        if self.tokenizer_type == "bart":
            end_token_index = torch.where(labels == self.tokenizer.eos_token_id)[1]
        else:
            end_token_index = torch.where(labels == self.tokenizer.sep_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx + 1:] = -100  # cross entropy cal

        data = {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': labels
        }
        if self.do_eval_only:
            data.update({
                'answer': ans,
                'question': qes,
                'context': ctx,
                'ans_span': ans_spans,
                'idx': idxs,
                'is_impossible': imp
            })

        if self.print_example:
            print(x_trunc)
            print(y_trunc)
            self.print_example = False

        return data


class BARTFinetuneModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--min_learning_rate', default=1e-7, type=float)
        parser.add_argument('--lr_decay_steps', default=0, type=int)
        parser.add_argument('--lr_decay_ratio', default=1.0, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup_steps', default=1000, type=int)
        parser.add_argument('--warmup_ratio', default=0.01, type=float)
        parser.add_argument('--label_smooth', default=0, type=float)
        parser.add_argument('--new_token_path', default="./", type=str)  # save new token after add special token
        parser.add_argument('--adam_beta1', default=0.9, type=float)
        parser.add_argument('--adam_beta2', default=0.999, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--scheduler_type', default='polynomial', type=str)

        return parent_args

    def __init__(self, tokenizer, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = BartForConditionalGeneration.from_pretrained(args.model_path)
        self.tokenizer = tokenizer

        # add special token ans
        # self.tokenizer.save_vocabulary(self.args.model_path)
        new_vocab = args.model_path+"/sp_vocab/"
        if not os.path.exists(new_vocab):
            os.makedirs(new_vocab)
        self.tokenizer.save_pretrained(new_vocab)
        self.model.resize_token_embeddings(len(tokenizer))
        self.vocab_size = len(tokenizer)
        self.rougescore = ROUGEScore(rouge_keys=('rougeL'), normalizer=lambda x: x)

        if self.hparams.label_smooth:
            self.loss_fct = LabelSmoothingCrossEntropy(smoothing=0.1)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
                self.total_steps = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        return configure_optimizers(self)

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        loss = output.loss
        if self.hparams.label_smooth:
            loss = self.loss_fct(output.logits.view(-1, self.vocab_size), batch["labels"].view(-1))

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        acc = self.compute_acc(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)
        self.log('val_ppl', torch.exp(output.loss), sync_dist=True)

        cond_output = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            do_sample=True,
            num_beams=5,
            early_stopping=True,
            max_length=64,
            top_p=0.9,
        )

        batch_label = torch.where(batch["labels"] != -100, batch["labels"], self.tokenizer.pad_token_id)
        pred = self.tokenizer.batch_decode(cond_output, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        ques = self.tokenizer.batch_decode(batch_label, clean_up_tokenization_spaces=True, skip_special_tokens=True)

        pred = [chinese_char_tokenize(white_space_fix(p)) for p in pred]
        ques = [chinese_char_tokenize(white_space_fix(q)) for q in ques]
        self.rougescore.update(pred, ques)

        return pred

    def validation_epoch_end(self, validation_step_outputs):
        rouge = self.rougescore.compute()
        self.log('val_rouge', rouge["rougeL_fmeasure"], sync_dist=True)

    def on_predict_start(self):
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def predict_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        loss_tensor = self.loss_fct(output.logits.transpose(1, 2), batch["labels"])
        if self.hparams.tokenizer_type == 'bart':
            eos_index = torch.where(batch['labels'] == self.tokenizer.eos_token_id)[1]
        elif self.hparams.tokenizer_type == 'bert':
            eos_index = torch.where(batch['labels'] == self.tokenizer.sep_token_id)[1]

        loss = torch.sum(loss_tensor, dim=1) / eos_index

        with torch.no_grad():
            cond_output = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                do_sample=True,
                num_beams=5,
                max_length=64,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True
            )

        pred = self.tokenizer.batch_decode(
            cond_output.sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)  # ['sequences']
        pred = [white_space_fix(p) for p in pred]  # remove prompt and white space
        score = cond_output.sequences_scores
        return pred, score, loss

    def compute_acc(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/y_true.shape[0]
        return acc

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


def get_tokenizer(tokenizer_type, pretrained_model_path):
    if tokenizer_type == 'bart':
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path, use_fast=False, additional_special_tokens=["<ans>", "<beg>", "<end>"])
        print(len(tokenizer))
    elif tokenizer_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_path, use_fast=False, additional_special_tokens=["[ANS]"])
    return tokenizer


def main():
    total_parser = argparse.ArgumentParser("Finetune BART for QG")
    total_parser.add_argument('--do_eval_only', action='store_true', default=False)
    total_parser.add_argument('--tokenizer_type', type=str, default="bart", choices=['bart', 'bert'])
    total_parser.add_argument('--tensorboard_dir', type=str, default="bart")
    total_parser.add_argument('--deepspeed')

    total_parser = UniversalDataModule.add_data_specific_args(total_parser)
    total_parser = QGT5Collator.add_data_specific_args(total_parser)
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
    total_parser = BARTFinetuneModel.add_model_specific_args(total_parser)
    args = total_parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer_type, args.model_path)
    collator = QGT5Collator(tokenizer=tokenizer, args=args)
    data_model = UniversalDataModule(collate_fn=collator, tokenizer=tokenizer, args=args)
    print("Data load complete...")

    if args.deepspeed is not None:
        os.environ['PL_DEEPSPEED_CONFIG_PATH'] = args.deepspeed

    model = BARTFinetuneModel(tokenizer, args)
    checkpoint_callback = UniversalCheckpoint(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[checkpoint_callback, lr_monitor]
                                         )

    if not args.do_eval_only:
        trainer.fit(model, data_model)


if __name__ == '__main__':
    main()

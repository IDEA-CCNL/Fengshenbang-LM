from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import LearningRateMonitor
from dataclasses import dataclass
import os
import argparse
import torch
import math
import time
from torch.utils.data._utils.collate import default_collate
from fengshen.data.data_utils.mask_utils import create_masked_lm_predictions
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils import UniversalCheckpoint
from fengshen.models.model_utils import (
    get_total_steps,
    configure_optimizers,
    add_module_args,
)
import numpy as np
SHOW_DATA = False


@ dataclass
class BartCollator:
    '''
    由input处理成samples，也就是最终模型的输入
    其中主要处理逻辑在__call__里
    包含text infilling和sentence shuffle任务
    '''
    tokenizer: None  # 分词
    max_seq_length: 512
    masked_lm_prob: 0.15
    permute_sentence_ratio: 1.0
    content_key: str = 'text'

    def setup(self):
        from fengshen.data.data_utils.sentence_split import ChineseSentenceSplitter
        self.sentence_split = ChineseSentenceSplitter()
        self.np_rng = np.random.RandomState(seed=((int(time.time()) % 2**32)))
        inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.vocab_id_list = list(inv_vocab.keys())
        self.vocab_id_to_token_dict = inv_vocab
        import jieba_fast
        self.zh_tokenizer = jieba_fast.lcut
        seg_tokens = ['。', ';', '；', '!', '！', '?', '？']
        seg_token_ids = []
        for t in seg_tokens:
            if t in self.tokenizer.vocab:
                seg_token_ids.append(self.tokenizer.vocab[t])
            else:
                print('seg_token "{}" not in vocab'.format(t))
        self.seg_token_ids = set(seg_token_ids)

    def permute_sentences(self, source, full_stops, p=1.0):
        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1): sentence_ends[i]]
            result[index: index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    def __call__(self, samples):
        '''
        samples: 一个sample长这样{"text": "hello world"}
        '''
        model_inputs = []
        for s in samples:
            sentences = self.sentence_split.tokenize(s[self.content_key])
            tokenized_sentences = [self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sent)) for sent in sentences]
            if len(tokenized_sentences) == 0:
                print('find empty sentence')
                continue

            tokens = [self.tokenizer.cls_token_id]
            for sent in tokenized_sentences:
                for t in sent:
                    tokens.append(t)
            if tokens[-1] != self.tokenizer.sep_token_id:
                tokens.append(self.tokenizer.sep_token_id)

            if len(tokens) > self.max_seq_length:
                # 找到最后的一句话，如果有的话，尽量保证最后一句话的完整
                last_pos = self.max_seq_length - 1
                for i in range(self.max_seq_length - 1, 0, -1):
                    if tokens[i-1] in self.seg_token_ids:
                        last_pos = i
                        break
                tokens = tokens[:last_pos]

                tokens.append(self.tokenizer.sep_token_id)
            tokens = torch.LongTensor(tokens)

            full_stops = torch.any(torch.stack([torch.eq(tokens, aelem).logical_or_(
                torch.eq(tokens, aelem)) for aelem in self.seg_token_ids], dim=0), dim=0)

            assert (self.max_seq_length -
                    tokens.shape[0]) >= 0, (tokens.size(), tokens[-1], self.max_seq_length)

            source, target = tokens, tokens.clone()

            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, full_stops, self.permute_sentence_ratio)

            if self.masked_lm_prob > 0.0:
                mask_prob = self.masked_lm_prob * 2
                max_predictions_per_seq = mask_prob * len(source)
                (source, _, _, _, _) = create_masked_lm_predictions(
                    source.numpy(), self.vocab_id_list, self.vocab_id_to_token_dict, mask_prob,
                    self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id,
                    max_predictions_per_seq, self.np_rng,
                    masking_style='bert', zh_tokenizer=self.zh_tokenizer)
                # 合并[MASK] 因为这里用的是Bert的mask函数，Bert是按字mask的，
                # 这里把连续的mask合并成一个MASK从而达到span mask的效果
                span_mask_souce = []
                for t in source:
                    # 如果是连续的多个mask，则跳过
                    if len(span_mask_souce) > 0 \
                            and t is self.tokenizer.mask_token_id \
                            and span_mask_souce[-1] is self.tokenizer.mask_token_id:
                        continue
                    span_mask_souce.append(t)

                source = torch.LongTensor(span_mask_souce)

            assert (source >= 0).all()
            # assert (source[1:-1] >= 1).all(), source
            assert (source <= self.tokenizer.vocab_size).all()
            assert source[0] == self.tokenizer.cls_token_id
            assert source[-1] == self.tokenizer.sep_token_id

            prev_output_tokens = torch.zeros_like(target)
            # match the preprocessing in fairseq
            prev_output_tokens[0] = self.tokenizer.sep_token_id
            prev_output_tokens[1:] = target[:-1]

            source_ = torch.full((self.max_seq_length,),
                                 self.tokenizer.pad_token_id, dtype=torch.long)
            source_[:source.shape[0]] = source
            target_ = torch.full((self.max_seq_length,), -100, dtype=torch.long)
            target_[:target.shape[0]] = target
            prev_output_tokens_ = torch.full(
                (self.max_seq_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            prev_output_tokens_[:prev_output_tokens.shape[0]] = prev_output_tokens
            attention_mask = torch.full((self.max_seq_length,), 0, dtype=torch.long)
            attention_mask[:source.shape[0]] = 1
            model_inputs.append({
                "input_ids": source_,
                "labels": target_,
                "decoder_input_ids": prev_output_tokens_,
                "attention_mask": attention_mask,
            })
        return default_collate(model_inputs)


class RandengBart(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Randeng BART')
        parser.add_argument('--masked_lm_prob', type=float, default=0.15)
        parser.add_argument('--max_seq_length', type=int, default=512)
        parser.add_argument('--sample_content_key', type=str, default='text')
        parser.add_argument('--permute_sentence_ratio', type=str, default=1.0)
        return parent_parser

    def __init__(self, args, tokenizer, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        config = BartConfig.from_pretrained(args.model_path)
        self.model = BartForConditionalGeneration(config)
        self.tokenizer = tokenizer

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def detokenize(self, token_ids):
        toks = self.tokenizer.convert_ids_to_tokens(token_ids)
        return self.tokenizer.convert_tokens_to_string(toks)

    def training_step(self, batch, batch_idx):
        if self.trainer.global_rank == 0:
            global SHOW_DATA
            if not SHOW_DATA:
                SHOW_DATA = True
                print('source: {}'.format(batch['input_ids'][0]))
                print('target: {}'.format(batch['labels'][0]))
                print('decoder source: {}'.format(batch['decoder_input_ids'][0]))

                print('source: {}'.format(self.detokenize(batch['input_ids'][0])))
                print('decoder source: {}'.format(self.detokenize(batch['decoder_input_ids'][0])))
                label_idx = batch['labels'][0] != -100
                print('target: {}'.format(self.detokenize(
                    batch['labels'][0][label_idx])))
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss, sync_dist=True)
        self.log('train_acc', acc, sync_dist=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        label_idx = labels != -100
        labels = labels[label_idx]
        logits = logits[label_idx].view(-1, logits.size(-1))
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.shape[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = RandengBart.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    collator = BartCollator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        masked_lm_prob=args.masked_lm_prob,
        content_key=args.sample_content_key,
        permute_sentence_ratio=args.permute_sentence_ratio,
    )
    # 准备一些额外参数
    collator.setup()
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collator)

    module = RandengBart(args, tokenizer=tokenizer)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    # 做兼容，如果目录不存在的话把这个参数去掉，不然会报错
    if args.load_ckpt_path is not None and \
            not os.path.exists(args.load_ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.load_ckpt_path = None

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(module, data_module, ckpt_path=args.load_ckpt_path)

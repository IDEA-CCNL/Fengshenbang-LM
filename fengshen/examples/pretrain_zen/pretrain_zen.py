from dataclasses import dataclass
from transformers import (
    BertTokenizer,
)
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
import argparse
import torch
import os
import numpy as np
import time
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.data.data_utils.sop_utils import get_a_and_b_segments
from fengshen.data.data_utils.truncate_utils import truncate_segments
from fengshen.data.data_utils.token_type_utils import create_tokens_and_tokentypes
from fengshen.data.data_utils.mask_utils import create_masked_lm_predictions
from fengshen.data.data_utils.ngram_utils import create_ngrams
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.models.zen3.configuration_zen import BertConfig as ZenConfig
from fengshen.models.zen3.modeling_zen import BertForPreTraining as ZenForPreTraining
from fengshen.models.zen2.ngram_utils import ZenNgramDict
from torch.utils.data._utils.collate import default_collate

SHOW_DATA = False


@dataclass
class ZenCollator:
    '''
    由input处理成samples，也就是最终模型的输入
    其中主要处理逻辑在__call__里
    包含Mask和Sop任务
    '''
    tokenizer: None  # 分词
    ngram_dict: None
    max_seq_length: 512
    max_ngram_length: 128
    masked_lm_prob: 0.15
    content_key: str = 'sentence'
    # 一些预处理操作

    def setup(self):
        import jieba_fast
        from fengshen.data.data_utils.sentence_split import ChineseSentenceSplitter
        self.sentence_split = ChineseSentenceSplitter()
        self.np_rng = np.random.RandomState(seed=((int(time.time()) % 2**32)))
        inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.vocab_id_list = list(inv_vocab.keys())
        self.vocab_id_to_token_dict = inv_vocab
        self.zh_tokenizer = jieba_fast.lcut

    def __call__(self, samples):
        '''
        samples: 一个sample长这样{"text": "hello world"}
        '''
        # start_time=time.time()
        model_inputs = []
        for s in samples:
            sentences = self.sentence_split.tokenize(s[self.content_key])
            # Divide sample into two segments (A and B).
            tokenized_sentences = [self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(sent)) for sent in sentences]
            if len(tokenized_sentences) == 0:
                print('find empty sentence')
                continue
            if len(tokenized_sentences) > 1:
                tokens_a, tokens_b, is_next_random = get_a_and_b_segments(tokenized_sentences,
                                                                          self.np_rng)
            else:
                tokens_a = tokenized_sentences[0]
                tokens_b = []
                is_next_random = False
            # max_seq_length - 3因为还需要拼上[CLS] [SEP] [SEP]
            if len(tokens_a) == 0:
                continue
            _ = truncate_segments(tokens_a, tokens_b, len(tokens_a),
                                  len(tokens_b), self.max_seq_length-3, self.np_rng)

            # Build tokens and toketypes.
            tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b,
                                                              self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)

            # Masking.
            max_predictions_per_seq = self.masked_lm_prob * len(tokens)
            (tokens, masked_positions, masked_labels, _, _) = create_masked_lm_predictions(
                tokens, self.vocab_id_list, self.vocab_id_to_token_dict, self.masked_lm_prob,
                self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id,
                max_predictions_per_seq, self.np_rng,
                masking_style='bert',
                zh_tokenizer=self.zh_tokenizer)
            
            # Ngram.
            ngram_ids, ngram_starts, ngram_lengths, ngram_freqs = create_ngrams(tokens, self.tokenizer, self.ngram_dict)
            ngram_ids, ngram_starts, ngram_lengths, ngram_freqs = ngram_ids[:self.max_ngram_length], ngram_starts[:self.max_ngram_length], ngram_lengths[:self.max_ngram_length], ngram_freqs[:self.max_ngram_length]
            ngram_ids_num = len(ngram_ids)
            ngram_positions_matrix = torch.zeros(size=(self.max_seq_length, self.max_ngram_length), dtype=float)
            for i in range(ngram_ids_num):
                ngram_positions_matrix[ngram_starts[i]:ngram_starts[i] + ngram_lengths[i], i] = ngram_freqs[i]
            ngram_positions_matrix = torch.div(ngram_positions_matrix, torch.stack([torch.sum(ngram_positions_matrix, 1)] * ngram_positions_matrix.size(1)).t() + 1e-10)

            # Some checks.
            num_tokens = len(tokens)
            num_ngrams = len(ngram_ids)
            padding_length = self.max_seq_length - num_tokens
            padding_ngram_length = self.max_ngram_length - num_ngrams
            assert padding_ngram_length >= 0
            assert padding_length >= 0
            assert len(tokentypes) == num_tokens
            assert len(masked_positions) == len(masked_labels)

            # Tokens and token types.
            filler = [self.tokenizer.pad_token_id] * padding_length
            tokens_np = np.array(tokens + filler, dtype=np.int64)
            tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

            filter_ngrams = [self.ngram_dict.ngram_to_id_dict['[pad]']] * padding_ngram_length
            ngrams_np = np.array(ngram_ids + filter_ngrams, dtype=np.int64)

            # Padding mask.
            padding_mask_np = np.array([1] * num_tokens + [0] * padding_length, dtype=np.int64)
            padding_ngram_mask_np = np.array([1] * num_ngrams + [0] * padding_ngram_length, dtype=np.int64)

            # Lables and loss mask.
            labels = [-100] * self.max_seq_length
            for i in range(len(masked_positions)):
                assert masked_positions[i] < num_tokens
                labels[masked_positions[i]] = masked_labels[i]
            labels_np = np.array(labels, dtype=np.int64)

            model_inputs.append(
                {
                    'input_ids': tokens_np,
                    'input_ngram_ids': ngrams_np,
                    'attention_mask': padding_mask_np,
                    'ngram_attention_mask': padding_ngram_mask_np,
                    'ngram_position_matrix': ngram_positions_matrix,
                    'token_type_ids': tokentypes_np,
                    'labels': labels_np,
                    'next_sentence_label': int(is_next_random)
                }
            )
            # end_time=time.time()
            # print("time for data process: "+str(end_time-start_time))
        return default_collate(model_inputs)


class ZEN(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Erlangshen Bert')
        parser.add_argument('--masked_lm_prob', type=float, default=0.15)
        parser.add_argument('--max_seq_length', type=int, default=512)
        parser.add_argument('--max_ngram_length', type=int, default=128)
        parser.add_argument('--sample_content_key', type=str, default='text')
        return parent_parser

    def __init__(self, args, tokenizer, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        config = ZenConfig.from_pretrained(args.model_path)
        self.config = config
        self.tokenizer = tokenizer
        self.model = ZenForPreTraining(config=config)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        return configure_optimizers(self)

    def forward(self, **batch):
        return self.model(**batch)

    def detokenize(self, token_ids):
        toks = self.tokenizer.convert_ids_to_tokens(token_ids)
        return self.tokenizer.convert_tokens_to_string(toks)

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.shape[0]
        return acc

    def training_step(self, batch, batch_idx):
        if self.trainer.global_rank == 0:
            global SHOW_DATA
            if not SHOW_DATA:
                print(self.config)
                print(self.model)
                SHOW_DATA = True
                print('source: {}'.format(batch['input_ids'][0]))
                print('target: {}'.format(batch['labels'][0]))
                print('source: {}'.format(self.detokenize(batch['input_ids'][0])))
                label_idx = batch['labels'][0] != -100
                print('target: {}'.format(self.detokenize(
                    batch['labels'][0][label_idx])))
        # start_time=time.time()
        output = self(**batch)
        # end_time=time.time()
        # print("time for model training: "+str(end_time-start_time))

        self.log('train_loss', output.loss, sync_dist=True)
        label_idx = batch['labels'] != -100
        acc = self.comput_metrix(
            output.prediction_logits[label_idx].view(-1, output.prediction_logits.size(-1)), batch['labels'][label_idx])
        self.log('train_acc', acc, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.log('val_loss', output.loss, sync_dist=True)
        return output.loss

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
    args_parser = ZEN.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    ngram_dict = ZenNgramDict(args.model_path, tokenizer=tokenizer)
    collate_fn = ZenCollator(
        tokenizer=tokenizer,
        ngram_dict=ngram_dict,
        max_seq_length=args.max_seq_length,
        max_ngram_length=args.max_ngram_length,
        masked_lm_prob=args.masked_lm_prob,
        content_key=args.sample_content_key,
    )
    collate_fn.setup()
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    print('data load complete')

    model = ZEN(args, tokenizer=tokenizer)
    print('model load complete')

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

    trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)

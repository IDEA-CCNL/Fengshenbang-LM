from dataclasses import dataclass
from transformers import (
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    AutoTokenizer,
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
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.data.data_utils.truncate_utils import truncate_segments
from fengshen.data.data_utils.token_type_utils import create_tokens_and_tokentypes
from fengshen.data.data_utils.mask_utils import create_masked_lm_predictions
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from torch.utils.data._utils.collate import default_collate

SHOW_DATA = False


@dataclass
class DeBERTaV2Collator:
    '''
    由input处理成samples，也就是最终模型的输入
    其中主要处理逻辑在__call__里
    包含Mask任务，使用Whole Word Mask
    '''
    tokenizer: None  # 分词
    max_seq_length: 512
    masked_lm_prob: 0.15
    content_key: str = 'text'
    # 一些预处理操作

    def setup(self):
        self.np_rng = np.random.RandomState(seed=42)
        inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.vocab_id_list = list(inv_vocab.keys())
        self.vocab_id_to_token_dict = inv_vocab
        import jieba_fast
        self.zh_tokenizer = jieba_fast.lcut

    def __call__(self, samples):
        '''
        samples: 一个sample长这样{"text": "hello world"}
        '''
        model_inputs = []
        for s in samples:
            tokenized_sentences = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(s[self.content_key]))
            if len(tokenized_sentences) == 0:
                print('find empty sentence')
                continue
            tokens_a = tokenized_sentences
            # max_seq_length - 3因为还需要拼上[CLS] [SEP] [SEP]
            if len(tokens_a) == 0:
                continue
            _ = truncate_segments(tokens_a, [], len(tokens_a),
                                  0, self.max_seq_length-3, self.np_rng)
            # Build tokens and toketypes.
            tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, [],
                                                              self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
            # Masking.
            max_predictions_per_seq = self.masked_lm_prob * len(tokens)
            (tokens, masked_positions, masked_labels, _, _) = create_masked_lm_predictions(
                tokens, self.vocab_id_list, self.vocab_id_to_token_dict, self.masked_lm_prob,
                self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id,
                max_predictions_per_seq, self.np_rng,
                masking_style='bert',
                zh_tokenizer=self.zh_tokenizer)

            # Some checks.
            num_tokens = len(tokens)
            padding_length = self.max_seq_length - num_tokens
            assert padding_length >= 0
            assert len(tokentypes) == num_tokens
            assert len(masked_positions) == len(masked_labels)

            # Tokens and token types.
            filler = [self.tokenizer.pad_token_id] * padding_length
            tokens_np = np.array(tokens + filler, dtype=np.int64)
            tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

            # Padding mask.
            padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                                       dtype=np.int64)

            # Lables and loss mask.
            labels = [-100] * self.max_seq_length
            for i in range(len(masked_positions)):
                assert masked_positions[i] < num_tokens
                labels[masked_positions[i]] = masked_labels[i]
            labels_np = np.array(labels, dtype=np.int64)
            model_inputs.append(
                {
                    'input_ids': tokens_np,
                    'attention_mask': padding_mask_np,
                    'token_type_ids': tokentypes_np,
                    'labels': labels_np,
                }
            )
        return default_collate(model_inputs)


class ErlangshenDeBERTaV2(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Erlangshen Bert')
        parser.add_argument('--masked_lm_prob', type=float, default=0.15)
        parser.add_argument('--max_seq_length', type=int, default=512)
        parser.add_argument('--sample_content_key', type=str, default='text')
        return parent_parser

    def __init__(self, args, tokenizer, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        config = DebertaV2Config.from_pretrained(args.model_path)
        self.config = config
        self.tokenizer = tokenizer
        self.model = DebertaV2ForMaskedLM(config)

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
        output = self(**batch)
        self.log('train_loss', output.loss, sync_dist=True)
        label_idx = batch['labels'] != -100
        acc = self.comput_metrix(
            output.logits[label_idx].view(-1, output.logits.size(-1)), batch['labels'][label_idx])
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
    args_parser = ErlangshenDeBERTaV2.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    collate_fn = DeBERTaV2Collator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        masked_lm_prob=args.masked_lm_prob,
        content_key=args.sample_content_key,
    )
    collate_fn.setup()
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    print('data load complete')

    model = ErlangshenDeBERTaV2(args, tokenizer=tokenizer)
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

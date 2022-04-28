import time
from fengshen.data.t5_dataloader.t5_datasets import UnsuperviseT5DataModel
from builtins import print
import sys
import os
import torch
import argparse
import json
import pytorch_lightning as pl
from transformers import MT5Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, loggers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import MT5ForConditionalGeneration
sys.path.append('../../../')
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'


class MT5PretrainModelCheckpoint:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('MT5PretrainModelCheckpoint')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./ckpt/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)
        parser.add_argument('--save_last', action='store_true', default=True)
        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=100, type=float)
        parser.add_argument('--save_weights_only', default=True, type=bool)

        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         every_n_train_steps=args.every_n_train_steps,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.dirpath,
                                         filename=args.filename,
                                         save_last=args.save_last)


class MT5PretrainModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        parser.add_argument('--keep_tokens_path', default=None, type=str)
        return parent_args

    def __init__(self, args, num_data):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.num_data = num_data
        keep_tokens = json.load(open(args.keep_tokens_path))
        print('num_data:', num_data)
        self.model = MT5ForConditionalGeneration.from_pretrained(
            args.pretrained_model_path)
        new_config = self.model.config
        new_config.vocab_size = len(keep_tokens)
        print('vocab_size:', new_config.vocab_size)

        new_state_dict = self.model.state_dict()
        select_index = torch.tensor(keep_tokens)
        new_state_dict['encoder.embed_tokens.weight'] = torch.index_select(
            new_state_dict['encoder.embed_tokens.weight'], dim=0, index=select_index)
        new_state_dict['shared.weight'] = torch.index_select(
            new_state_dict['shared.weight'], dim=0, index=select_index)
        new_state_dict['decoder.embed_tokens.weight'] = torch.index_select(
            new_state_dict['decoder.embed_tokens.weight'], dim=0, index=select_index)
        new_state_dict['lm_head.weight'] = torch.index_select(
            new_state_dict['lm_head.weight'], dim=0, index=select_index)
        self.model = MT5ForConditionalGeneration.from_pretrained(
            args.pretrained_model_path, config=new_config, state_dict=new_state_dict)
        # print(self.model)

    def setup(self, stage) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data /
                                  (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        # print('is out of index: ', batch['input_ids'][batch['input_ids'] >= 32598])
        output = self.model(
            input_ids=batch['input_ids'], labels=batch['labels'])
        self.log('val_loss', output.loss)

    def predict_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'], labels=batch['labels'])
        print('predict_loss', output.loss)
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
        )
        return {'predict_ids': generated_ids, 'labels': batch['labels'], 'input_ids': batch['input_ids']}

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(paras, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.args.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def main():
    total_parser = argparse.ArgumentParser("Pretrain Unsupervise.")
    total_parser.add_argument(
        '--do_eval_only', action='store_true', default=False)
    # total_parser.add_argument('--ckpt_path', default=None, type=str)
    total_parser.add_argument(
        '--pretrained_model_path', default='/cognitive_comp/ganruyi/hf_models/google/mt5-small', type=str)
    total_parser.add_argument(
        '--new_vocab_path', default='/cognitive_comp/ganruyi/hf_models/t5_cn_small/sentencepiece_cn.model', type=str)
    total_parser.add_argument('--ckpt_path', default=None, type=str)
    total_parser.add_argument('--max_seq_length', default=1024, type=int)
    # * Args for data preprocessing
    total_parser = UnsuperviseT5DataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = MT5PretrainModelCheckpoint.add_argparse_args(total_parser)
    total_parser = MT5PretrainModel.add_model_specific_args(total_parser)
    # * Args for base model
    args = total_parser.parse_args()
    print('Argument parse success.')
    print('UnsuperviseT5DataModel load start {}'.format(get_time_str()))
    data_model = UnsuperviseT5DataModel(args)
    print('UnsuperviseT5DataModel load end {}'.format(get_time_str()))
    if not args.do_eval_only:
        model = MT5PretrainModel(args, len(data_model.train_dataloader()))
        checkpoint_callback = MT5PretrainModelCheckpoint(args).callbacks
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(
            args.default_root_dir, 'log/'), name='t5_cn_small_pretrain')
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             callbacks=[checkpoint_callback]
                                             )

        trainer.fit(model, data_model)
        model.model.save_pretrained(os.path.join(args.default_root_dir), 'hf_model')
        # trainer.fit(model, data_model, ckpt_path=args.ckpt_path)
    else:
        tokenizer = MT5Tokenizer.from_pretrained(args.new_vocab_path, extra_ids=0)
        model = MT5PretrainModel(args=args, num_data=len(data_model.predict_dataloader()))
        trainer = Trainer.from_argparse_args(args)

        result = trainer.predict(model, data_model)
        result = result[0]
        for i in range(4):
            print(tokenizer.batch_decode(result['input_ids'][i]))
            print(tokenizer.batch_decode(result['predict_ids'][i]))
            print(tokenizer.batch_decode(result['labels'][i]))


def demo(model_path):
    print('result from {}'.format(model_path))
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MT5Tokenizer.from_pretrained(model_path)
    model.to('cuda')
    model.eval()
    # print(tokenizer.encode('<extra_id_0>'))
    text = '纽约大学坐落于 <extra_id_0>，它有悠久的 <extra_id_1>。'
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to('cuda')
    ids = model.generate(input_ids, max_length=64)
    print('outputs:', tokenizer.decode(ids[0]))


def save():
    tokenizer = MT5Tokenizer.from_pretrained(
        '/cognitive_comp/ganruyi/experiments/t5_cn_small_pretrain/Randeng-T5-77M/sentencepiece_cn.model', extra_ids=0)
    tokenizer.save_pretrained('/cognitive_comp/ganruyi/experiments/t5_cn_small_pretrain/Randeng-T5-77M')


if __name__ == '__main__':
    # main()
    cn_model_path = '/cognitive_comp/ganruyi/experiments/t5_cn_small_pretrain_v2/Randeng-T5-77M'
    model_path = '/cognitive_comp/ganruyi/hf_models/google/mt5-small'
    demo(cn_model_path)
    demo(model_path)
    # save()

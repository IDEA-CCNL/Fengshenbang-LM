import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = '3,5'
import torch
import argparse

import pytorch_lightning as pl

from dataclasses import dataclass

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from pytorch_lightning import (
    Trainer,
)

from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from transformers import (
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config)
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from torch.utils.data import default_collate


@dataclass
class GPTDataCollator:
    
    tokenizer: None  # 分词
    max_seq_length: 1024
    content_key: str = 'text'

    def setup(self):
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __call__(self,samples):
        # samples: {'text':[doc1,doc2,doc...,batchsize]}
        # print(samples)
        # print(samples[self.content_key])
        samples = default_collate(samples)
        inputs = self.tokenizer.batch_encode_plus(samples[self.content_key],max_length=self.max_seq_length,padding='max_length',truncation=True,return_tensors='pt',add_special_tokens=False)
        labels = inputs['input_ids'].clone().detach()
        labels[inputs['input_ids'] == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        # ”“’‘''""
        # inputs:
        # {
        #   'input_ids': tensor([[19526,   254, 25001,   121, 28938,   245,   171,   120,   253, 50256]],[input_ids2],....batchsize]), 
        #   'attention_mask': tensor([[am1],[am2],......batchsize]), 
        #   'labels': tensor([[19526,   254, 25001,   121, 28938,   245,   171,   120,   253,  -100],[labels2],.....batchsize])
        # }
        # print('========== show data ============')
        # # print(f'^_^ :',samples)
        # print(f'^_^ samples : {samples["text"][0]}')
        # print(f'^_^ ids : {inputs["input_ids"][0]}')
        # print(f'^_^ deocde ids : {self.tokenizer.decode(inputs["input_ids"][0])}')
        return inputs


def _cut_sent(para):
    """
    split doc into sentence
    """
    para += ' '
    
    # 匹配 \1: 句子结束符  \2: 1个非’”，也就是不被引号包裹的句子
    para = re.sub('([？。！\?\!…]+)([^”’]|[”’])', r'\1#####\2', para)
    para = re.sub('([\.]{3,})([^”’])', r'\1#####\2', para)

    # 匹配 \1: 句子结束符紧挨’”  \2: 非句子结束符号，被引号包裹的句子
    para = re.sub('([。！？\?\!…][”’])([^，。！？\?\!]|\s)', r'\1#####\2', para)
    para = re.sub('([\.]{3,}[”’])([^，。！？\?\!]|\s)', r'\1#####\2', para)
    para = re.sub('([#]{5})([”’])([^，。！？\?\!])', r'\2#####\3', para)
    para = para.strip()
    
    # 将多个\n拼接的也分开
    para = re.sub('[\n]+','#####',para)

    return [s.strip() for s in para.split("#####") if s]


def map_fun(samples,content_key='text'):
    """
    pad every sample's length <= 1024
    """
    chunks = []
    for doc in samples[content_key]:
        line = ''
        for para in re.split('[\n]+',doc):
            for sentence in _cut_sent(para):
                if len(line) <= 1024:
                    if len(line+sentence) >= 1024:
                        if len(line) == 0:
                            chunks.append(sentence)
                            line = ''
                        else:
                            chunks.append(line)
                            line = sentence
                    else:
                        line += sentence
            line += '\n'
        chunks.append(line.strip())
    return {content_key:chunks}


class GPT2Model(pl.LightningModule):

    @staticmethod
    def add_module_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--max_seq_length', type=int, default=1024)
        parser.add_argument('--sample_content_key', type=str, default='text')
        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        # if train from a tempty model use this:
        self.config = GPT2Config.from_pretrained(args.model_path)
        self.model = GPT2LMHeadModel(self.config)
        # else use this
        # self.model = GPT2LMHeadModel.from_pretrained(args.model_path)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print(f'batch size: {self.hparams.train_batchsize}')
            print(f'world size: {self.trainer.world_size}')
            print(f'accumulate_grad_batches: {self.trainer.accumulate_grad_batches}')
            print('Total steps: {}' .format(self.total_steps))

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        # output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss,sync_dist=True)
        self.log('train_acc', acc, sync_dist=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        # 计算acc 的时候，label要往后移一位才正确；模型计算loss的时候自动内置了这个操作
        # 去掉pad的label，算的更准确
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred[:,:-1].reshape((-1,))
        y_true = labels[:,1:].reshape((-1,)).float()
        # 去掉pad的label，算的更准确
        # y_true = y_true_[y_true_ != -100]
        # y_pred = y_pred[y_true_ != -100]
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/corr.shape[0]
        return acc

    def validation_step(self, batch, batch_idx):
        # print('now doing validation',batch_idx)
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss,sync_dist=True)
        self.log('val_acc', acc,sync_dist=True)

    def configure_optimizers(self):
        return configure_optimizers(self)


def main():
    args_parser = argparse.ArgumentParser("gpt pretrain")
    args_parser = add_module_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = GPT2Model.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()
    print('args parse done')
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    collate_fn = GPTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        content_key=args.sample_content_key,
    )
    collate_fn.setup()
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    # 将数据 分句拼接 变成1024长度的样本，如果数据不需要这样处理，可以忽略，如需要可以开启
    # print('---------mapping data---------')
    # data_module.datasets = data_module.datasets.map(map_fun,batched=True,
    #         num_proc=args.dataloader_workers,
    #         remove_columns=data_module.datasets.column_names['train'],
    #         load_from_cache_file=True)
    # print(data_module.datasets)
    # print('   ------check data------   ')
    # for i in range(10):
    #     print(f"train data {i}:\n{data_module.datasets['train'][i]}")
    print('data load done')
    model = GPT2Model(args, tokenizer=tokenizer)
    print('model init done')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    if args.load_ckpt_path is not None and not os.path.exists(args.load_ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.load_ckpt_path = None
    #limit_train_batches=1,limit_val_batches=1,limit_test_batches=1, 值为整数时，每个epoch🈯只过指定个数的batchse数
    trainer = Trainer.from_argparse_args(args,limit_train_batches=0.1,limit_val_batches=0.1,limit_test_batches=0.1,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)


if __name__ == '__main__':
    main()

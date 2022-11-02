# sys.path.append('./')
import os
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
from transformers import GPT2LMHeadModel,GPT2Tokenizer
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint


@dataclass
class GPTDataCollator:
    
    tokenizer: None  # 分词
    max_seq_length: 1024
    content_key: str = 'text'

    def setup(self):
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    def __call__(self,samples):
        # samples: {'text':[doc1,doc2,doc...,batchsize]}
        inputs = self.tokenizer.batch_encode_plus(samples[self.content_key],max_length=self.max_seq_lenght,padding='max_length',truncation=True,return_tensors='pt')
        labels = inputs['input_ids'].clone().detach()
        labels[inputs['input_ids'] == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        # inputs:
        # {
        #   'input_ids': tensor([[19526,   254, 25001,   121, 28938,   245,   171,   120,   253, 50256]],[input_ids2],....batchsize]), 
        #   'attention_mask': tensor([[am1],[am2],......batchsize]), 
        #   'labels': tensor([[19526,   254, 25001,   121, 28938,   245,   171,   120,   253,  -100],[labels2],.....batchsize])
        # }
        return inputs


class GPT2Model(pl.LightningModule):

    @staticmethod
    def add_module_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        # parser.add_argument('--learning_rate', default=1e-4, type=float)
        # parser.add_argument('--weight_decay', default=0.1, type=float)
        # parser.add_argument('--warmup', default=0.01, type=float)
        parser.add_argument('--max_seq_length', type=int, default=1024)
        parser.add_argument('--sample_content_key', type=str, default='text')
        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        # self.num_data = num_data
        # print('num_data:', num_data)
        self.model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        # output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss)
        self.log('train_acc', acc, sync_dist=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.shape[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss)
        self.log('val_acc', acc)

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

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    collate_fn = GPTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        content_key=args.sample_content_key,
    )
    collate_fn.setup()
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    model = GPT2Model(args, tokenizer=tokenizer)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    if args.load_ckpt_path is not None and not os.path.exists(args.load_ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.load_ckpt_path = None
    
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)

    # if not args.do_eval_only:
    #     model = GPT2Model(args, tokenizer=tokenizer)
    #     checkpoint_callback = GPT2Checkpoint(args).callbacks
    #     logger = loggers.TensorBoardLogger(save_dir=os.path.join(
    #         args.default_root_dir, 'log/'), name='WenZhong')
    #     trainer = Trainer.from_argparse_args(args,
    #                                          logger=logger,
    #                                          callbacks=[checkpoint_callback]
    #                                          )
    #     trainer.fit(model, data_model)


if __name__ == '__main__':
    main()

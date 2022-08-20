from dataclasses import dataclass
from transformers import (
    MegatronBertForPreTraining,
)
from pytorch_lightning import (
    LightningModule,
    loggers,
    Trainer,
)
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from typing import Optional
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
import argparse
import sys
import torch
import os

'''
由input处理成samples，也就是最终模型的输入
其中主要处理逻辑在__call__里
包含Mask和Sop任务
'''
@dataclass
class ErLangShenCollator:
    # 一些预处理操作
    def setup():
        
        return
    
    def __call__(self, samples):
        '''
        samples: 一个sample长这样{"text": "hello world"}
        '''
        
    


class ErLangShen(LightningModule):
    
    


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = MegatronDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = MegatronBertLGTN.add_module_specific_args(args_parser)
    args_parser = CustomCKPT.add_argparse_args(args_parser)
    args_parser.add_argument('--deepspeed')
    args = args_parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    data_module = MegatronDataModule(tokenizer, args)
    print('data load complete')

    model = MegatronBertLGTN(args)
    print('model load complete')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'),
        name=os.path.basename(os.path.dirname(args.model_path)))
    checkpoint_callback = CustomCKPT(args).callbacks

    if args.resume_from_checkpoint is not None and \
            not os.path.exists(args.resume_from_checkpoint):
        print('--------warning no checkpoint found--------, remove args')
        del args.resume_from_checkpoint

    # autotuning
    if args.deepspeed is not None:
        os.environ['PL_DEEPSPEED_CONFIG_PATH'] = args.deepspeed

    trainer = Trainer.from_argparse_args(args, logger=logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, data_module)

# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : base_trainer.py
#   Last Modified : 2022-04-13 17:52
#   Describe      : 
#
# ====================================================
import os
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin


class BaseTrainer:
    @staticmethod
    def add_trainer_specific_args(parent_parser):
        """
        Add trainer specific args
        Args:
            patience(int, optional): monitors metric on epoch end and stops training, if metric does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 3
            save_dir (str, optional): root save directory
            Additional args from pytorch-lightning Trainer:
                max_epoch(int, optional): max number of training epochs for each corruption on the dataset
                gpu_nums(int, optional): Number of gpus used for multi-gpu training. Set to 0 to disable gpus.
                precision(int, optional): Precision of float used in training. 16 or 32.
                strategy(str, optional): Supports different training strategies with aliases
                    as well custom training type plugins.
                etc.
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('BaseTrainer')
        # * Args for trainer setting
        parser.add_argument('--patience', default=-1, type=int)
        parser.add_argument('--save_dir', default='./outputs', type=str)
        parser.add_argument('--save_top_k', default=-1, type=int)
        parser.add_argument('--monitor', default='val_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--timestamp', default=None, type=str)
        parser.add_argument('--train', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--predict', action='store_true', default=False)

        parent_parser = pl.Trainer.add_argparse_args(parent_parser=parent_parser)

        return parent_parser

    def __init__(self, args, model) -> None:
        """
        initiates a Seq2SeqTrainer class, defines training procedures for seq2seq model like T5
        Args:
            args: contain trainer and callback parameters
            model: seq2seq model to train
        """
        self.model = model
        callbacks = [TQDMProgressBar(refresh_rate=1)]
        lr_callback = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_callback)
        checkpoint = ModelCheckpoint(dirpath=args.save_dir,
                                     save_top_k=args.save_top_k,
                                     save_last=True,
                                     monitor=args.monitor,
                                     mode=args.mode,
                                     filename='{epoch:02d}-{' + args.monitor + ':.4f}',
                                     every_n_train_steps=500,
                                     )
        #  checkpoint.CHECKPOINT_NAME_LAST = 'last-{epoch:02d}-{' + args.monitor + ':.4f}'
        callbacks.append(checkpoint)

        if args.patience > 0:
            early_stop_callback = EarlyStopping(
                monitor=args.monitor,
                min_delta=0.00,
                patience=args.patience,
                verbose=True,
                mode=args.mode,
                check_on_train_epoch_end=True,  # Check early stopping after every train epoch, ignore multi validation in one train epoch
            )
            callbacks.append(early_stop_callback)

        logger = loggers.TensorBoardLogger(save_dir=os.path.join(args.save_dir, 'logs/'), name="default")
        if args.strategy == "ddp":
            strategy = DDPPlugin(find_unused_parameters=False)
        else:
            strategy = args.strategy
        self.trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger if args.train else False,
            callbacks=callbacks,
            strategy=strategy,
            #  prepare_data_per_node=True,
        )

    def train(self, data_model: pl.LightningDataModule, **kwargs):
        """
        Train seq2seq model with given data model.
        Args:
            data_model: lightning data module
        """
        # Train
        self.trainer.fit(self.model, datamodule=data_model, **kwargs)

    def test(self, data_model: pl.LightningDataModule, **kwargs):
        # Test
        self.trainer.test(self.model, datamodule=data_model, **kwargs)

    def predict(self, data_model: pl.LightningDataModule, **kwargs):
        # Predict
        return self.trainer.predict(self.model, datamodule=data_model, **kwargs)


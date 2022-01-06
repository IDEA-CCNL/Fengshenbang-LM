from typing import Optional

import torch
from torch import nn
import time
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Dict, List, Optional, Tuple, Union

import fengshen.data.megatron_dataloader.bert_dataset as bert_dataset
from fengshen.trainer.cnnl_args import CNNLTrainningArguments


class CNNLTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None),
        cnnl_args: CNNLTrainningArguments = None
    ):
        assert cnnl_args != None, "cnnl args is none"
        Trainer.__init__(self,
                         model=model,
                         args=args,
                         tokenizer=tokenizer,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers)
        if torch.distributed.get_rank() == 0:
            start_time = time.time()
            print('> compiling dataset index builder ...')
            from fengshen.data.megatron_dataloader.dataset_utils import compile_helper
            compile_helper()
            print('>>> done with dataset index builder. Compilation time: {:.3f} '
                  'seconds'.format(time.time() - start_time), flush=True)
        bert_dataset.tokenizer = tokenizer
        self.train_dataset, self.eval_dataset = bert_dataset.build_train_valid_test_datasets(
            data_prefix=cnnl_args.megatron_data_path,
            data_impl=cnnl_args.megatron_data_impl,
            splits_string=cnnl_args.megatron_splits_string,
            train_valid_test_num_samples=None,
            max_seq_length=512,
            masked_lm_prob=0.15,
            short_seq_prob=0.1,
            seed=cnnl_args.megatron_seed,
            skip_warmup=False,
            binary_head=cnnl_args.megatron_binary_head)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            shuffle=True,
        )
        print("sampler %d", len(train_sampler))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=train_sampler,
        )

        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            self.eval_dataset,
            shuffle=True,
        )

        self.eval_loader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            sampler=eval_sampler,
        )
        self.test_dataset = self.eval_dataset

    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.eval_sampler

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.eval_sampler

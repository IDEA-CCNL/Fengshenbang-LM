from pytorch_lightning import Trainer, loggers, LightningModule
from timeit import default_timer as timer
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.models.gpt.configuration_gpt2 import GPT2Config
from fengshen.models.gpt.modeling_gpt2 import GPT2LMHeadModel
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import AutoTokenizer
from dataclasses import dataclass
import os
import re
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
import torch
import argparse
import logging
import traceback
import sys
sys.path.append('../../../')


# from transformers import GPT2LMHeadModel


@dataclass
class GPT2Collator:
    def __init__(self, tokenizer, max_seq_length=1024, content_key="text"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.content_key = content_key
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __call__(self, samples):
        # samples: {'text':[doc1,doc2,doc...,batchsize]}
        # print(samples)
        # print(samples[self.content_key])
        samples = {self.content_key: [s[self.content_key] for s in samples]}
        # samples = default_collate(samples)
        # 各种类型的数据[s1,s2..] 变成带批次的张量[tensor(s1),tensor(s2)] default_collate([(0, 1), (2, 3)])，替代 for s in samples, append 等操作
        inputs = self.tokenizer.batch_encode_plus(
            samples[self.content_key],
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = inputs['input_ids'].clone().detach()
        labels[inputs['input_ids'] == self.tokenizer.pad_token_id] = -100

        if torch.any(torch.isnan(inputs["input_ids"])):
            print("data has nan")

        return {
            'input_ids': inputs["input_ids"],
            'attention_mask': inputs["attention_mask"],
            'labels': labels
        }


def _cut_sent(para):
    """
    split doc into sentence
    """
    para += ' '

    # 匹配 \1: 句子结束符  \2: 1个非’”，也就是不被引号包裹的句子
    para = re.sub(r'([？。！\?\!…]+)([^”’]|[”’])', r'\1#####\2', para)
    para = re.sub(r'([\.]{3,})([^”’])', r'\1#####\2', para)

    # 匹配 \1: 句子结束符紧挨’”  \2: 非句子结束符号，被引号包裹的句子
    para = re.sub(r'([。！？\?\!…][”’])([^，。！？\?\!]|\s)', r'\1#####\2', para)
    para = re.sub(r'([\.]{3,}[”’])([^，。！？\?\!]|\s)', r'\1#####\2', para)
    para = re.sub(r'([#]{5})([”’])([^，。！？\?\!])', r'\2#####\3', para)
    para = para.strip()

    # 将多个\n拼接的也分开
    para = re.sub(r'[\n]+', '#####', para)

    return [s.strip() for s in para.split("#####") if s]


def _pad_max_length(samples, content_key='text', max_seq_len=2048):
    """
    pad every sample's length <= 1024
    """
    chunks = []
    for doc in samples[content_key]:
        line = ''
        for para in re.split('[\n]+', doc):
            for sentence in _cut_sent(para):
                if len(line) <= max_seq_len:
                    if len(line+sentence) >= max_seq_len:
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
    return {content_key: chunks}


class GPT2Trainer(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--max_seq_length', type=int, default=1024)
        parser.add_argument('--sample_content_key', type=str, default='text')
        parser.add_argument('--tensorboard_dir', type=str, default='gpt')
        parser.add_argument('--from_scratch', type=int, default=1)
        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        if self.hparams.from_scratch:
            self.config = GPT2Config.from_pretrained(args.model_path)
            self.model = GPT2LMHeadModel(self.config)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(args.model_path)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print(f'batch size: {self.hparams.train_batchsize}')
            print(f'world size: {self.trainer.world_size}')
            print(f'model Parameters: {sum(p.numel() for p in self.model.parameters()) / 1000000}M')
            print(f'accumulate_grad_batches: {self.trainer.accumulate_grad_batches}')
            print('Total steps: {}' .format(self.total_steps))

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss, sync_dist=True)
        self.log('train_acc', acc, sync_dist=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred[:, :-1].reshape((-1,))
        y_true = labels[:, 1:].reshape((-1,)).float()  # shift_right label auto in training steps loss
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/corr.shape[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def timer_setup(self):
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.timings = []

    def test_step(self, batch, batch_idx):
        # self.starter.record()
        # output = self.model.generate(
        #     input_ids=batch['input_ids'],
        #     attention_mask=batch['attention_mask'],
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     max_length=self.hparams.max_seq_length,
        #     do_sample=False,
        #     # num_beams=5,
        #     # temperature = 1.1,
        #     # top_k=20,
        #     # top_p=0.9,
        #     # repetition_penalty=1.1,
        #     # no_repeat_ngram_size = 2,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     # max_new_tokens=64,
        # )

        # # output = self.model(**batch)
        # self.ender.record()
        # torch.cuda.synchronize()
        # curr_time = self.starter.elapsed_time(self.ender)
        # self.timings.append(curr_time)

        self.starter.record()
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
            outputs = self.model(**batch)
        self.ender.record()
        # torch.cuda.synchronize()
        self.timings.append(self.starter.elapsed_time(self.ender))
        print(prof.table())
        return outputs

    def configure_optimizers(self):
        return configure_optimizers(self)


def main():
    args_parser = argparse.ArgumentParser("gpt pretrain")
    args_parser = add_module_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = GPT2Trainer.add_module_specific_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    # ! debug code
    logging.basicConfig(filename=os.path.join(args.default_root_dir, 'traceback2.log'),
                        level=logging.INFO, filemode='a',
                        format='[%(asctime)s] [%(levelname)s] >>>  %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S')
    logging.info(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    collate_fn = GPT2Collator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        content_key=args.sample_content_key,
    )

    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    data_module.datasets = data_module.datasets.map(
        _pad_max_length,
        batched=True,
        num_proc=args.dataloader_workers,
        remove_columns=data_module.datasets.column_names['train'],
        load_from_cache_file=True
    )
    print(data_module.datasets)
    for i in range(5):
        logging.info(f"train data {i}:\n{data_module.datasets['train'][i]}")
    logging.info("Data loaded")

    module = GPT2Trainer(args, tokenizer=tokenizer)
    logging.info("Model load")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    if args.load_ckpt_path is not None and not os.path.exists(args.load_ckpt_path):
        args.load_ckpt_path = None
        logging.warning("didn't find any ckpt")

    logger = loggers.TensorBoardLogger(save_dir=args.tensorboard_dir, name="tf_log")
    trainer = Trainer.from_argparse_args(
        args,
        # limit_train_batches=0.01,
        # limit_val_batches=1.0,
        # limit_test_batches=0.01,
        logger=logger,
        callbacks=[
            lr_monitor,
            checkpoint_callback
        ])

    start = timer()
    trainer.fit(module, data_module, ckpt_path=args.load_ckpt_path)
    end = timer()
    print(f"Time elapsed {end-start} s")

    # module.timer_setup()
    # trainer.test(module, data_module, ckpt_path=args.load_ckpt_path)
    # print(f"Train timing {sum(module.timings)/len(module.timings)} ms")


if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        logging.error("Main program error:")
        logging.error(e)
        logging.error(traceback.format_exc())

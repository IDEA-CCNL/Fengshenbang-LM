from asyncio.log import logger
from cgitb import lookup
from dataclasses import dataclass
import os
import deepspeed
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import argparse
from fengshen.models.model_utils import (
    configure_optimizers,
    add_module_args,
    get_total_steps
)
from fengshen.models.llama.modeling_llama import LlamaForCausalLM
from fengshen.models.megatron import mpu
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.strategies.megatron_deepspeed import DeepSpeedStrategy
from transformers import LlamaTokenizer
from llama_generate import generate
SHOW_DATA = False


def pad(ids, pad_id, max_length):
    if len(ids) > max_length:
        return ids[:max_length]
    return ids + [pad_id] * (max_length - len(ids))


prompt_prefix = ""
prompt_without_output = "<human>:{prompt}\n<bot>:"

@dataclass
class LlamaSFTCollator:
    '''
    由input处理成samples，也就是最终模型的输入
    其中主要处理逻辑在__call__里
    '''
    tokenizer: None  # 分词
    max_seq_length: 1536
    def __call__(self, samples):
        input_ids_list = []
        labels_list = []
        max_length = 0
        for s in samples:
            """
            sample: {
                "task" : str,
                "prompt": [str]
                "output": [str]
                }
            """
            prompt_cnt = min(len(s["prompt"]), len(s["output"]))
            # input_ids = self.tokenizer(prompt_prefix).input_ids
            input_ids = []
            labels_ids = [-100] * len(input_ids)
            for i in range(prompt_cnt):
                prompt_input_ids = self.tokenizer(prompt_without_output.format_map(
                    {"prompt": s["prompt"][i].strip()}), add_special_tokens=False).input_ids
                output_ids = self.tokenizer(s["output"][i].strip(), add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                
                input_ids += prompt_input_ids
                input_ids += output_ids
                
                labels_ids += [-100] * (len(prompt_input_ids)) + output_ids
            
            # input_ids += [self.tokenizer.eos_token_id]
            # labels_ids += [self.tokenizer.eos_token_id]
            max_length = min(max(len(input_ids), max_length), self.max_seq_length)
            input_ids_list.append(input_ids)
            labels_list.append(labels_ids)

        # PAD
        for i in range(len(input_ids_list)):
            labels_list[i] = pad(labels_list[i], -100, max_length)
            input_ids_list[i] = pad(input_ids_list[i], self.tokenizer.eos_token_id, max_length)
        model_inputs = {
            'input_ids': torch.tensor(input_ids_list).clone(),
            'attention_mask': torch.ones((len(input_ids_list), max_length)).clone(),
            "position_ids": torch.arange(0, max_length).unsqueeze(0).expand(len(input_ids_list), max_length).clone(),
            'labels': torch.tensor(labels_list).clone(),
        }
        return model_inputs

        
class Llama(pl.LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ziya_llama finetune')
        parser.add_argument('--max_seq_length', type=int, default=1024)
        parser.add_argument('--model_parallel_size', type=int, default=1)
        parser.add_argument('--tokenizer_path', default=None, type=str)
        return parent_parser

    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer

    def setup(self, stage) -> None:
        if mpu.get_model_parallel_world_size() > 1:
            self.model = LlamaForCausalLM.from_pretrained(
                f"{self.hparams.model_path}/part_{mpu.get_model_parallel_rank()}", torch_dtype=torch.half).cuda()
        else:
            self.model = LlamaForCausalLM.from_pretrained(f"{self.hparams.model_path}", torch_dtype=torch.half).cuda()
        
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}'.format(self.total_steps))


    def configure_optimizers(self):
        return configure_optimizers(self)

    def forward(self, **batch):
        return self.model(**batch)

    def detokenize(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def comput_metrix(self, logits, labels):
        with torch.no_grad():
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
                SHOW_DATA = True
                print('source: {}'.format(batch['input_ids'][0]))
                print('target: {}'.format(batch['labels'][0]))
                print('source: {}'.format(self.detokenize(batch['input_ids'][0])))
                label_idx = batch['labels'][0] != -100
                print('target: {}'.format(self.detokenize(
                    batch['labels'][0][label_idx])))
                print('mask: {}'.format(batch['attention_mask'][0]))
                print('position_ids: {}'.format(batch['position_ids'][0]))
        output = self(**batch)
        self.log('train/loss', output.loss, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.log('val_loss', output.loss, sync_dist=True)
        return output.loss

    def predict_step(self, batch, batch_idx):
        # generate data
        generate_kwargs = {
        	"do_sample": True,
        	"top_p": 1.0,   
        	"top_k": 0,
        	"max_length": 256,
        	"repetition_penalty": 1.0,
        	"temperature": 0.8,
        	"pad_token_id": self.tokenizer.eos_token_id,
        	"eos_token_id": self.tokenizer.eos_token_id,
        }
        batch_input_ids = batch['input_ids'].cpu().numpy().tolist()
        print('batch_input_ids:\n', batch_input_ids)
        queries = [self.detokenize(input_ids).split('<bot>:')[0].replace('<s>', '')+'<bot>:' for input_ids in batch_input_ids]
        print('queries:\n', queries)
        # queries = ['怎样给世界一点爱？', '生命的意义是什么？']
        ans = generate(queries=queries,
                tokenizer=self.tokenizer,
                model=self.model,
                device=self.model.device,
                **generate_kwargs)
        print('ans:\n', ans)
        ## end

    def on_load_checkpoint(self, checkpoint) -> None:
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--do_eval_only', action='store_true', default=False)
    args_parser.add_argument('--wandb_project', type=str, default="ziya_llama13b_finetune_example")
    args_parser.add_argument('--wandb_name', type=str, default="exp1")
    args_parser = add_module_args(args_parser)
    args_parser = pl.Trainer.add_argparse_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Llama.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    collate_fn = LlamaSFTCollator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    print('data load complete')
    model = Llama(args, tokenizer=tokenizer)
    print('model load complete')
    print(model)

    strategy = DeepSpeedStrategy(
        tensor_model_parallel_size=args.model_parallel_size,
        pipe_model_parallel_size=1,
        mpu_seed=42,
    )
    if args.load_ckpt_path is not None and \
            not os.path.exists(args.load_ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.load_ckpt_path = None
    if not args.do_eval_only:
        wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_name)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = UniversalCheckpoint(args)
        
        trainer = pl.Trainer.from_argparse_args(
            args, 
            strategy=strategy,
            logger=wandb_logger,
            callbacks=[lr_monitor, checkpoint_callback])
        trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)
    else:
        trainer = pl.Trainer.from_argparse_args(args, strategy=strategy)
        trainer.predict(model, data_module, ckpt_path=args.load_ckpt_path)

from dataclasses import dataclass
from pytorch_lightning import (
    LightningModule,
    Trainer
)
from pytorch_lightning.loggers import WandbLogger
from transformers import LlamaTokenizer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    
)

import argparse
import torch
import os
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.models.llama.modeling_llama import LlamaForCausalLM
from fengshen.strategies.megatron_deepspeed import DeepSpeedStrategy

SHOW_DATA = False


def pad(ids, pad_id, max_length):
    if len(ids) > max_length:
        return ids[:max_length]
    return ids + [pad_id] * (max_length - len(ids))


prompt_prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
prompt_without_output = "\n### Instruction: {prompt} \n### Response:"
#prompt_prefix = ""
#prompt_without_output = "\n### [问]: {prompt}\n### [子牙]:"

@dataclass
class LlamaSFTCollator:
    '''
    由input处理成samples，也就是最终模型的输入
    其中主要处理逻辑在__call__里
    '''
    tokenizer: None  # 分词
    max_seq_length: 1536
    enter_token_id = 13
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
            input_ids = self.tokenizer(prompt_prefix).input_ids
            labels_ids = [-100] * len(input_ids)
            for i in range(prompt_cnt):
                # 补上一个换行符
                # input_ids += [self.enter_token_id]
                # labels_ids += [-100]
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
            input_ids_list[i] = pad(input_ids_list[i], self.tokenizer.pad_token_id, max_length)
        model_inputs = {
            'input_ids': torch.tensor(input_ids_list).clone(),
            'attention_mask': torch.ones((len(input_ids_list), max_length)).clone(),
            "position_ids": torch.arange(0, max_length).unsqueeze(0).expand(len(input_ids_list), max_length).clone(),
            'labels': torch.tensor(labels_list).clone(),
        }
        return model_inputs


class Llama(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('llama sft')
        parser.add_argument('--max_seq_length', type=int, default=1024)
        return parent_parser

    def __init__(self, args, tokenizer, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer

    def setup(self, stage) -> None:
        self.model = LlamaForCausalLM.from_pretrained(
            self.hparams.model_path, torch_dtype=torch.half).cuda()
        print(f"resize embedding {len(self.tokenizer)}")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.enable_lora()
        self.model.config.use_cache = False
        self.config = self.model.config

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
        output = self(**batch)
        self.log('train/loss', output.loss, sync_dist=True)
        # label_idx = batch['labels'][:, 1:] != -100
        # acc = self.comput_metrix(
        #    # right shift
        #    output.logits[:, :-1, :][label_idx].view(-1, output.logits.size(-1)),
        #    batch['labels'][:, 1:][label_idx])
        # self.log('train/acc', acc, sync_dist=True, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.log('val_loss', output.loss, sync_dist=True)
        return output.loss

    def on_load_checkpoint(self, checkpoint) -> None:
        # self.consumed_samples = 664576
        # self.trainer.fit_loop.epoch_progress.current.reset()
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = Llama.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    collate_fn = LlamaSFTCollator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    print('data load complete')

    model = Llama(args, tokenizer=tokenizer)
    print('model load complete')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    # 做兼容，如果目录不存在的话把这个参数去掉，不然会报错
    if args.load_ckpt_path is not None and \
            not os.path.exists(args.load_ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.load_ckpt_path = None

    strategy = DeepSpeedStrategy(
        pipe_model_parallel_size=1,
        tensor_model_parallel_size=1,
        mpu_seed=1234
    )

    wandb_logger = WandbLogger(project="llama_13b_plus_sft")  # 初始化个WandbLogger对象
    trainer = Trainer.from_argparse_args(args,
                                         strategy=strategy,
                                         logger=wandb_logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)

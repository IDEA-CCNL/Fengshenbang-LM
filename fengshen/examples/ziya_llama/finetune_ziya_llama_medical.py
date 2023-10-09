from dataclasses import dataclass
import os
import deepspeed
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import argparse
from fengshen.models.model_utils import (
    configure_optimizers,
    add_module_args,
    get_total_steps
)
# from fengshen.models.llama.modeling_llama import LlamaForCausalLM
from fengshen.models.megatron import mpu
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.strategies.megatron_deepspeed import DeepSpeedStrategy
from transformers import LlamaTokenizer, LlamaForCausalLM
from fengshen.utils.utils import chinese_char_tokenize
from llama_generate import generate
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import corpus_bleu
SHOW_DATA = False


def generate_sentence(raw_list):
    words = []
    i = 0
    while i < len(raw_list) and raw_list[i] != "</s>":
        words.append(raw_list[i])
        i += 1
    return "".join(words)


def remove_pad(raw_text, ref=False):
    if ref:
        return [raw_text.lstrip("<pad>")]
    else:
        return raw_text.lstrip("<pad>")


def compute_bleu(preditions, labels):
    score_nltk = corpus_bleu(labels, preditions)
    return score_nltk


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


@dataclass
class LlamaInferenceCollator:
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
            labels_ids = []
            for i in range(prompt_cnt):
                prompt_input_ids = self.tokenizer(prompt_without_output.format_map(
                    {"prompt": s["prompt"][i].strip()}), add_special_tokens=False).input_ids
                output_ids = self.tokenizer(s["output"][i].strip(), add_special_tokens=False).input_ids
                
                input_ids += prompt_input_ids
                labels_ids += output_ids
            
            # input_ids += [self.tokenizer.eos_token_id]
            # labels_ids += [self.tokenizer.eos_token_id]
            max_length = min(max(len(labels_ids),max(len(input_ids), max_length)), self.max_seq_length)
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
        parser.add_argument("--prediction_res_path", default=None, type=str)
        return parent_parser

    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        self.rouge_metric = ROUGEScore(
            rouge_keys=("rougeL", "rouge1", "rouge2"), normalizer=lambda x: x
        )
        self.bleu_val = []

    def setup(self, stage) -> None:
        if mpu.get_model_parallel_world_size() > 1:
            self.model = LlamaForCausalLM.from_pretrained(
                f"{self.hparams.model_path}/part_{mpu.get_model_parallel_rank()}", torch_dtype=torch.half).cuda()
        else:
            self.model = LlamaForCausalLM.from_pretrained(f"{self.hparams.model_path}", torch_dtype=torch.bfloat16).cuda()
            self.model.gradient_checkpointing_enable()
        
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
        predict_ids = self.model.generate(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            **generate_kwargs)


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

        return {
            "input_ids": batch["input_ids"],
            "predict_ids": predict_ids,
            "labels": batch["labels"],
        }
        ## end

    
    def save_preditions(self, result, args):
        with open(args.prediction_res_path, "w", encoding="utf8") as fw:
            predictions = []
            labels = []

            predictions2 = []
            labels2 = []
            for batch in result:
                print(batch.keys())
                # print(batch['labels'])
                # use eos_id replace pad_id
                for i in range(len(batch["input_ids"])):
                    context = self.tokenizer.decode(
                        batch["input_ids"][i],
                        cleanup_tokenization_space=True,
                        skip_special_tokens=False,
                    )
                    context = context.split('<bot>')[0].replace('<s>','').replace('</s>','')

                    pred = self.tokenizer.decode(
                        batch["predict_ids"][i],
                        cleanup_tokenization_space=True,
                        skip_special_tokens=False,
                    )
                    pred = pred.split('<bot>')[-1].replace('<s>','').replace('</s>','')
                    label_idx = batch['labels'] != -100
                    target = self.detokenize(batch['labels'][label_idx])
                    
                    # self.rouge_metric.update(
                    #     preds=chinese_char_tokenize(pred).cuda(),
                    #     target=chinese_char_tokenize(target).cuda(),
                    # )
                    predictions.append(list(pred))
                    labels.append([list(target)])

                    predictions2.append(pred)
                    labels2.append(target)

                    fw.write("context:" + "".join(context) + "\n")
                    fw.write("pred:" + pred + "\n")
                    fw.write("target" + target + "\n")
                    fw.write("\n")
            bleu = compute_bleu(predictions, labels)
            fw.write("bleu:{}".format(bleu))
            # rouge_dict = self.rouge_metric.compute()
            rouge_dict = self.rouge_metric(predictions2, labels2)
            # reset the metric after once validation
            # self.rouge_metric.reset()
            for k, v in rouge_dict.items():
                fw.write(f"{k}:{v}\n")
        print("finish prediction, saved in {}".format(args.prediction_res_path))
        return predictions, labels



    def on_load_checkpoint(self, checkpoint) -> None:
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))
            self.tokenizer.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

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
    if not args.do_eval_only:
        collate_fn = LlamaSFTCollator(
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        collate_fn = LlamaInferenceCollator(
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
    # strategy = 'deepspeed_stage_2_offload'
    if args.load_ckpt_path is not None and \
            not os.path.exists(args.load_ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.load_ckpt_path = None
    if not args.do_eval_only:
        wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_name)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = UniversalCheckpoint(args)
        
        trainer = Trainer.from_argparse_args(
            args, 
            strategy=strategy,
            logger=wandb_logger,
            callbacks=[lr_monitor, checkpoint_callback])
        print(f'devices:{args.devices}, accelerator:{args.accelerator}')
        trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)
    else:
        trainer = Trainer.from_argparse_args(args, strategy=strategy)
        print(f'devices:{args.devices}, accelerator:{args.accelerator}')
        result = trainer.predict(model, data_module, ckpt_path=args.load_ckpt_path)
        predictions, labels = model.save_preditions(result, args)
        sample = result[0]  # first_batch
        batch_labels = torch.where(
            sample["labels"] != -100, sample["labels"], model.tokenizer.eos_token_id
        )
        for i in range(2):
            print(tokenizer.batch_decode(sample["input_ids"][i]))
            print(tokenizer.batch_decode(sample["predict_ids"][i]))
            print(tokenizer.batch_decode(batch_labels[i]))

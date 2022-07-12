from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_lightning import (
    LightningModule,
    Trainer,
    loggers,
)
from torchtext.data.metrics import bleu_score
from pytorch_lightning.callbacks import LearningRateMonitor
from dataclasses import dataclass
import os
import argparse
import torch, logging, traceback
import sys
sys.path.append('../../')

from fengshen.data.dusinc_dataloader import DusincDataModule
from fengshen.utils import UniversalCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# rewarite from pretrain_bart Text Filling Collator
# x = src + kno +  tgt
def truncate_input_sequence(document:str, max_num_tokens:int):
    total_length = len(document)
    if total_length <= max_num_tokens:
        return document
    else: 
        return document[:max_num_tokens]
    
def padding_to_maxlength(ids, max_length, pad_id):
    cur_len = len(ids)
    len_diff = max_length - len(ids)
    return ids + [pad_id] * len_diff, [1] * cur_len + [0] * len_diff
@dataclass
class DialoCollator:
    tokenizer: None
    max_seq_length: int = 512 
    max_kno_length: int = 256
    max_src_length: int = 128
    max_tgt_length: int = 128

    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Bart Text Filling Collator')
        parser.add_argument('--max_seq_length', default=512, type=int) #总序列最长多长
        parser.add_argument('--max_src_length', default=256, type=int) #总序列最长多长
        parser.add_argument('--max_kno_length', default=128, type=int) #知识最长多长
        parser.add_argument('--max_tgt_length', default=128, type=int) #回复最长多长
        return parent_args

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = args.max_seq_length
        
    def generate_sample(self, x):
        # tokenize sentence
        x = tokenizer.bos_token + x + tokenizer.eos_token
        input_dicts = tokenizer.encode_plus(
            x,
            max_length=self.max_seq_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'
        )
        
        input_ids = input_dicts["input_ids"]
        attn_mask = input_dicts["attention_mask"]
        labels = input_ids

        return [input_ids, labels, attn_mask]

    def __call__(self, samples):
        input_ids, labels, attn_mask = [],[],[]
        for s in samples:
            # 需要补充 bos , eos, 所以最长长度需要-2
            s["knowledge"] = truncate_input_sequence(s["knowledge"],self.args.max_kno_length-2)
            s["src"] = truncate_input_sequence(s["src"],self.args.max_src_length-2)
            s["tgt"] = truncate_input_sequence(s["tgt"],self.args.max_tgt_length-1)

            x_trunc = f'knowledge: {s["knowledge"]} context: {s["src"]} response:{s["tgt"]}' #prompt
         
            g = self.generate_sample(x_trunc)
            
            input_ids.append(g[0])
            labels.append(g[1])
            attn_mask.append(g[2])

        return {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': torch.cat(labels),
            "knowledge": s["knowledge"],
            "question":s["src"]
        }

@dataclass
class QueryCollator:
    tokenizer: None
    max_seq_length: int = 512 
    max_src_length: int = 496
    max_tgt_length: int = 16

    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Bart Text Filling Collator')
        parser.add_argument('--max_seq_length', default=512, type=int) #总序列最长多长
        parser.add_argument('--max_src_length', default=496, type=int) #总序列最长多长
        parser.add_argument('--max_tgt_length', default=16, type=int) #回复最长多长
        return parent_args

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = args.max_seq_length
        
    def generate_sample(self, x):
        # tokenize sentence
        x = tokenizer.bos_token + x + tokenizer.eos_token
        input_dicts = tokenizer.encode_plus(
            x,
            max_length=self.max_seq_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'
        )

        logging.info(x)
        logging.info(input_dicts["input_ids"])
        
        input_ids = input_dicts["input_ids"]
        attn_mask = input_dicts["attention_mask"]
        labels = input_ids

        return [input_ids, labels, attn_mask]

    def __call__(self, samples):
        input_ids, labels, attn_mask = [],[],[]
        for s in samples:
            # 需要补充 bos , eos, 所以最长长度需要-2
            s["src"] = truncate_input_sequence(s["src"],self.args.max_src_length-2)
            s["tgt"] = truncate_input_sequence(s["tgt"],self.args.max_tgt_length-1)

            x_trunc = f'context: {s["src"]} query:{s["tgt"]}' #prompt
         
            g = self.generate_sample(x_trunc)
            
            input_ids.append(g[0])
            labels.append(g[1])
            attn_mask.append(g[2])

        return {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': torch.cat(labels),
            "query": s["tgt"],
            "question":s["src"]
        }

class GPT2Finetuner(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Bart Lightning')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        return parent_parser

    def __init__(self, args, tokenizer, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.model = GPT2LMHeadModel.from_pretrained(args.model_path)
        self.tokenizer = tokenizer

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batchsize * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # generate & __call__ diff: https://huggingface.co/docs/transformers/v4.19.4/en/internal/generation_utils#transformers.generation_utils.GreedySearchDecoderOnlyOutput
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'])
        # output doc https://huggingface.co/docs/transformers/main_classes/output1
        # GPT models https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel.forward.returns
        self.log('train_loss', output.loss, sync_dist=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        """rewrite"""
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / labels.size()[0]
        return acc   

    def validation_step(self, batch, batch_idx):
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'])

        acc = self.comput_metrix(output.logits, batch["labels"])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

if __name__ == '__main__':

    
    try: 
        args_parser = argparse.ArgumentParser()
        args_parser = DusincDataModule.add_data_specific_args(args_parser)
        args_parser = Trainer.add_argparse_args(args_parser)
        args_parser = GPT2Finetuner.add_module_specific_args(args_parser)
        args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
        args_parser = DialoCollator.add_data_specific_args(args_parser)
        args_parser.add_argument('--deepspeed')
        args_parser.add_argument('--pretrain_sp_tokenizer', type=str,default='')
        args_parser.add_argument('--task', type=str,default='dial')
        args = args_parser.parse_args()

        # ! debug code
        logging.basicConfig(filename=os.path.join(args.default_root_dir,'traceback2.log'),
            level=logging.INFO, filemode='a', 
            format='[%(asctime)s] [%(levelname)s] >>>  %(message)s',
            datefmt='%Y-%m-%d %I:%M:%S')
        logging.debug(args)
        # ! debug code

        tokenizer = GPT2Tokenizer.from_pretrained(args.pretrain_sp_tokenizer,extra_ids=0)
        tokenizer.add_special_tokens({'pad_token': "[PAD]"}) #[PAD]


        # for task 2
        if args.task == "query":
            collator = QueryCollator(tokenizer, args)
        else:
            collator = DialoCollator(tokenizer, args)
        
        data_module = DusincDataModule(tokenizer=tokenizer, args=args, collate_fn=collator)
        logging.info("Data has been loader")

        module = GPT2Finetuner(args, tokenizer)
        logging.info("Finetuner has been loading")

        lr_monitor = LearningRateMonitor(logging_interval='step')
        logger = loggers.TensorBoardLogger(save_dir=args.default_root_dir,
            name=os.path.basename(os.path.dirname(args.model_path)))
        checkpoint_callback = UniversalCheckpoint(args).callbacks
        logging.info("Monitor and ckpt has been loaded")

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

        logging.info("Begin training")

        trainer.fit(module, data_module)
        module.model.save_pretrained(args.dirpath+'/hf_pretrained_model_'+args.task)
        
        #trainer.test(ckpt_path=trainer.checkpoint_callback.last_model_path,             dataloaders=data_module, verbose=True)
    except Exception as e:
        logging.error("Main program error:")
        logging.error(e)
        logging.error(traceback.format_exc())   
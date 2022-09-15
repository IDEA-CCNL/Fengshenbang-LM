from dataclasses import dataclass
from lib2to3.pgen2 import token
import time
import sys
import os    
import logging, traceback

from matplotlib.style import context
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from transformers import BartForConditionalGeneration, BertTokenizer, AutoTokenizer
from pytorch_lightning.callbacks import LearningRateMonitor
sys.path.append('../../../')
from fengshen.data.t5_dataloader.dialo_datasets import DialoT5DataModule
from fengshen.data.t5_dataloader.t5_datasets import TaskT5DataModel
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.data.universal_datamodule import UniversalDataModule

def truncate_sequence(document:str, max_num_tokens:int,reverse=False):
    total_length = len(document)
    if total_length <= max_num_tokens:
        return document
    else: 
        if reverse:
            return document[-1*max_num_tokens:]
        else:
            return document[:max_num_tokens]

def padding_to_maxlength(ids, max_length, pad_id):
    cur_len = len(ids)
    len_diff = max_length - len(ids)
    return ids + [pad_id] * len_diff, [1] * cur_len + [0] * len_diff

@dataclass
class DialoT5Collator:
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('BART DIalo Collator')
        parser.add_argument('--max_seq_length', default=512, type=int) #总序列最长多长
        parser.add_argument('--max_src_length', default=256, type=int) #总序列最长多长
        parser.add_argument('--max_kno_length', default=128, type=int) #知识最长多长
        parser.add_argument('--max_tgt_length', default=128, type=int) #回复最长多长
        return parent_args

    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        
    def encode(self, x, y):
        """
        参考 Unified QA 
        # https://github.com/allenai/unifiedqa/blob/master/bart/unified_data.py
        """
        # tokenize sentence
        x = self.tokenizer.bos_token + x + self.tokenizer.eos_token
        y = y + self.tokenizer.eos_token

        encoder_input = self.tokenizer.encode_plus(
            x,
            max_length=self.args.max_kno_length+ self.args.max_src_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'
        )
        decoder_output = self.tokenizer.encode_plus(
            y,
            max_length=self.args.max_tgt_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'       
        )
        return encoder_input, decoder_output

    def __call__(self, samples):
        for s in samples:
            s["knowledge"] = s["kno"] # 兼容不同数据集键名

        input_ids,  attn_mask, decoder_input_ids, decoder_attn_mask = [],[],[],[]
        for s in samples:
            # 需要补充 prompt(2) bos(1), eos(1)，所以最长长度 -3
            # bos prompt [kno] prompt [src] eos
            s["knowledge"] = truncate_sequence(s["knowledge"],self.args.max_kno_length-3)
            s["src"] = truncate_sequence(s["src"],self.args.max_src_length-3, reverse=True) # 倒叙截取上下文问句，以提升对最近问句的相应
            s["tgt"] = truncate_sequence(s["tgt"],self.args.max_tgt_length-1)#后面要加 eos

            x_trunc = f'knowledge: {s["knowledge"]} context: {s["src"]}' #prompt
            y_trunc = f'{s["tgt"]}'
            encoder_input, decoder_output = self.encode(x_trunc,y_trunc)
            
            input_ids.append(encoder_input["input_ids"])
            attn_mask.append(encoder_input["attention_mask"])
            decoder_input_ids.append(decoder_output["input_ids"])
            
            # TODO 是否需要 decoder attn mask 
            # 参考 BART Summary task_datasets
            #decoder_attn_mask.append(decoder_input[1])

        return {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': torch.cat(decoder_input_ids),
            'knowledge': s["knowledge"],
            'question':s["src"]
        }

@dataclass
class QGT5Collator:
    @ staticmethod
    def add_data_specific_args(parent_args):
        # the hyperparameters should be determined according to the max length of context in dataset
        parser = parent_args.add_argument_group('BART DIalo Collator')
        parser.add_argument('--max_seq_length', default=512, type=int) 
        parser.add_argument('--max_src_length', default=16, type=int) 
        parser.add_argument('--max_kno_length', default=432, type=int) 
        parser.add_argument('--max_tgt_length', default=64, type=int) 
        return parent_args

    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.print_example = True
        
    def encode(self, x, y):
        # tokenize sentence
        x = self.tokenizer.bos_token + x + self.tokenizer.eos_token
        y = y + self.tokenizer.eos_token

        encoder_input = self.tokenizer.encode_plus(
            x,
            max_length=self.args.max_kno_length+ self.args.max_src_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'
        )
        decoder_output = self.tokenizer.encode_plus(
            y,
            max_length=self.args.max_tgt_length, 
            padding="max_length",
            truncation=True, 
            return_tensors='pt'       
        )
        return encoder_input, decoder_output

    def __call__(self, samples):
        """
        参考 RGX https://github.com/luohongyin/RGX/blob/main/ques_gen_ft.py
        处理 Add Mask answer span, concat masked context and answer
        Input:
        input_ids: input_ids (text + answer)
        attn_mask: input attn mask
        labels:   decoder_ids (question)
        """
        input_ids,  attn_mask, decoder_input_ids, decoder_attn_mask = [],[],[],[]
        for s in samples:
            ans_bos, ans_eos = s["ans_span"]
            context = s["context"][:ans_bos] + self.tokenizer.mask_token + s["context"][ans_eos:]
            context = truncate_sequence(context,self.args.max_kno_length-1)
            answer = truncate_sequence(s["answer"],self.args.max_src_length-1) #src and tgt is reversed in qg
            question = truncate_sequence(s["question"],self.args.max_tgt_length-1) 
            
            x_trunc = f'{context} </s> {answer}' #prompt
            y_trunc = f'{question}'
            encoder_input, decoder_output = self.encode(x_trunc,y_trunc)

            if self.print_example:
                print(x_trunc)
                print(y_trunc)
                self.print_example = False

            input_ids.append(encoder_input["input_ids"])
            attn_mask.append(encoder_input["attention_mask"])
            decoder_input_ids.append(decoder_output["input_ids"])
        

        return {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': torch.cat(decoder_input_ids),
        }
    

class BARTFinetuneModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel') 
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)#config optimizer 
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        #parser.add_argument('--keep_tokens_path', default=None, type=str)
        return parent_args

    def __init__(self, tokenizer, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = BartForConditionalGeneration.from_pretrained(args.model_path)
        self.tokenizer = tokenizer

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
                self.total_steps = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])    
        self.log('train_loss', output.loss, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        # print('is out of index: ', batch['input_ids'][batch['input_ids'] >= 32598])
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        acc = self.compute_acc(output.logits, batch['labels'])
        # ppl = self.compute_ppl(output.loss, batch['labels'], batch['attention_mask'])
        # cond_output = self.model.generate(
        #     input_ids=batch['input_ids'],
        #     attention_mask=batch['attention_mask'],
        #     #force_words_ids=batch['force_words_ids'],
        #     num_beams=4,
        # )
        # The size of tensor a (8) must match the size of tensor b (1024) at non-singleton dimension 0
        # cond_acc = self.comput_metrix(cond_output, batch['labels'])
        # self.log('cond_acc', cond_acc, sync_dist=True)
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)
        # self.log('val_ppl', ppl, sync_dist=True)

    def compute_ppl(self, loss, labels, attn_mask):
        shift_attn_mask = attn_mask[:, 1:].contiguous()
        meanloss = loss.sum(1) / shift_attn_mask.sum(1)
        ppl = torch.exp(meanloss).numpy().tolist()
        return ppl

    def compute_acc(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/y_true.shape[0]
        return acc

    def on_save_checkpoint(self, checkpoint) -> None:
        # Save the current loop info in the mid of epoch
        # if you lightning <= 1.6.0  uncomment the line below
        # checkpoint['loops'] = self.trainer.checkpoint_connector._get_loops_state_dict()
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
        # if self.trainer.global_rank == 0 and self.trainer.global_step % self.hparams.every_n_train_steps == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset

def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def get_tokenizer(tokenizer_type, pretrained_model_path):
    if tokenizer_type == 'bart':
        return AutoTokenizer.from_pretrained(pretrained_model_path)
    else:
        return BertTokenizer.from_pretrained(pretrained_model_path)

def main():
    total_parser = argparse.ArgumentParser("Finetune BART for QG")
    total_parser.add_argument('--do_eval_only', action='store_true', default=False)
    total_parser.add_argument('--tokenizer_type', type=str, default="bart")
    total_parser.add_argument('--tensorboard_dir', type=str, default="bart")
    total_parser.add_argument('--deepspeed')

    # Args for data preprocessing
    # total_parser = TaskT5DataModel.add_data_specific_args(total_parser)
    total_parser = DialoT5DataModule.add_data_specific_args(total_parser)
    total_parser = QGT5Collator.add_data_specific_args(total_parser)

    # Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
    total_parser = BARTFinetuneModel.add_model_specific_args(total_parser)
    
    # Args for base model
    args = total_parser.parse_args()

    # ! debug code
    logging.basicConfig(filename=os.path.join(args.default_root_dir,'traceback2.log'),
        level=logging.INFO, filemode='a', 
        format='[%(asctime)s] [%(levelname)s] >>>  %(message)s',
        datefmt='%Y-%m-%d %I:%M:%S')
    logging.debug(args)

    #data_model = TaskT5DataModel(args)
    tokenizer = get_tokenizer(args.tokenizer_type, args.model_path)
    collator = QGT5Collator(tokenizer=tokenizer,args=args)
    #collator = DialoT5Collator(tokenizer=tokenizer,args=args)
    #data_model = UniversalDataModule(tokenizer=tokenizer,args=args,collate_fn=collator)
    data_model = DialoT5DataModule(collate_fn=collator,tokenizer=tokenizer, args=args)
    logging.info('Load QGT5 QA Data')
    
    if args.deepspeed is not None:
        os.environ['PL_DEEPSPEED_CONFIG_PATH'] = args.deepspeed

    if not args.do_eval_only:
        logging.info("Begin Training")
        model = BARTFinetuneModel(tokenizer,args)
        checkpoint_callback = UniversalCheckpoint(args).callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        logger = loggers.TensorBoardLogger(save_dir=args.tensorboard_dir, name="tf_log")
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             callbacks=[checkpoint_callback, lr_monitor]
                                             )
        trainer.fit(model, data_model)
    else:
        trainer = Trainer.from_argparse_args(args)
        model = BARTFinetuneModel(tokenizer,args)
        trainer.validate(model, data_model)


if __name__ == '__main__':
    try:
        main()
    except Exception as e: 
        logging.error("Main program error:")
        logging.error(e)
        logging.error(traceback.format_exc())   
from ast import arg
from sched import scheduler
import sys
import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers, seed_everything
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import get_scheduler

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np

from uie.extraction import constants
from uie.extraction.record_schema import RecordSchema
from uie.extraction.extraction_metrics import get_extract_metrics
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
from uie.extraction.dataset_processer import PrefixGenerator
from uie.seq2seq.data_collator import DynamicSSIGenerator

from uie.extraction.record_schema import RecordSchema
from uie.extraction.constants import BaseStructureMarker
from uie.extraction.utils import convert_to_record_function

from uie.seq2seq.features import RecordFeature
from uie.seq2seq.t5_bert_tokenizer import T5BertTokenizer
from loguru import logger

class UIEDataset(Dataset):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--dataset_num_workers', default=8, type=int)

        parser.add_argument('--max_source_length', default=512, type=int)
        parser.add_argument('--max_target_length', default=512, type=int)
        
        parser.add_argument('--ignore_pad_token_for_loss', default=True, type=bool)
        parser.add_argument('--pad_to_max_length', default=True, type=bool)
        
       
        parser.add_argument('--text_column', default='text', type=str)
        parser.add_argument('--record_column', default='record', type=str)
        parser.add_argument('--source_prefix', default='meta: ' , type=str)
        
        return parent_args
    
    def __init__(self, data_file, tokenizer, args, sample_prompt):
        super().__init__()
        
        self.args = args
        self.tokenizer =tokenizer        
        self.dataset_num_workers = args.dataset_num_workers
        self.prefix = self.add_prefix()
        self.data_file = {'dataset':data_file}
        self.sample_prompt = sample_prompt
        self.data = self.load_data()
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self):
        # from datasets import load_dataset
        datasets = load_dataset('etc/uie_json.py', data_files=self.data_file)
        
        logger.info('Source:{}, Target:{}'.format(self.args.text_column,self.args.record_column))
        logger.info(datasets)
     
        column_names = datasets['dataset'].column_names
        if self.sample_prompt:
            datasets = datasets['dataset'].map(self.tokenize_function, batched=True, num_proc=self.args.dataset_num_workers, remove_columns=column_names, features=RecordFeature)
        else:
            datasets = datasets['dataset'].map(self.tokenize_function_eval, batched=True, num_proc=self.args.dataset_num_workers, remove_columns=column_names, features=RecordFeature)

        return datasets

    def add_prefix(self):
        if self.args.record_schema and os.path.exists(self.args.record_schema):
            record_schema = RecordSchema.read_from_file(self.args.record_schema)
        else:
            record_schema = None
        if self.args.source_prefix is not None:
            if self.args.source_prefix == 'schema':
                prefix = PrefixGenerator.get_schema_prefix(schema=record_schema)
            elif self.args.source_prefix.startswith('meta'):
                prefix = ""
            else:
                prefix = self.args.source_prefix
        else:
            prefix = ""
        
        logger.info(f"Prefix: {prefix}")
        logger.info(f"Prefix Length: {len(self.tokenizer.tokenize(prefix))}")
        return prefix
        
    def tokenize_function(self, examples):
        inputs = examples[self.args.text_column]
        targets = examples[self.args.record_column]
        inputs = [self.prefix + inp for inp in inputs]
    
        padding = "max_length" if self.args.pad_to_max_length else False
        model_inputs = self.tokenizer(inputs, add_special_tokens=False, max_length=self.args.max_source_length,padding=padding, truncation=True)
    
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, add_special_tokens=False, max_length=self.args.max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and self.args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(_label if _label != self.tokenizer.pad_token_id else -100) for _label in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        if self.args.source_prefix is not None and self.args.source_prefix.startswith('meta'):
            model_inputs['spots'] = examples['spot']
            model_inputs['asocs'] = examples['asoc']
            model_inputs['spot_asoc'] = examples['spot_asoc']
            model_inputs['sample_prompt'] = [True] * len(model_inputs['input_ids'])   # sample_prompt=True for Finetune and Pretrain

        return model_inputs

    def tokenize_function_eval(self, examples):
        model_inputs = self.tokenize_function(examples)
        model_inputs['sample_prompt'] = [False]*len(model_inputs['input_ids'])
        return model_inputs


class UIEDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--max_length', default=512, type=int)
        parser.add_argument('--max_prefix_length', default=-1, type=int)
        
        parser.add_argument('--val_batchsize', default=16, type=int)
        
        parser.add_argument('--meta_positive_rate', default=1., type=float)
        parser.add_argument('--negative', default=-1, type=int)
        parser.add_argument('--ordered_prompt', default=False, type=bool)
        
        parser.add_argument('--decoding_format', default='spotasoc', type=str)
        parser.add_argument('--spot_noise', default=0, type=float, help='实体动态采样的比率')
        parser.add_argument('--asoc_noise', default=0, type=float, help="关系动态采样的比率")
        
        parser.add_argument('--padding', default=True, type=bool)
        parser.add_argument('--pad_to_multiple_of', default=None)
        parser.add_argument('--label_pad_token_id', default=-100, type=int)
        
        return parent_args

    def __init__(self, train_file, valid_file, test_file, tokenizer, args):
        super().__init__()
        self.args = args
        self.train_data = UIEDataset(train_file, tokenizer, args, sample_prompt=True)
        self.valid_data = UIEDataset(valid_file, tokenizer, args, sample_prompt=False)
        self.test_data = UIEDataset(test_file, tokenizer, args, sample_prompt=False)
        self.tokenizer = tokenizer
        self.schema =  RecordSchema.read_from_file(args.record_schema) #
        self.negative_sampler = self.dynamicSSIGenerator()
        self.spot_asoc_nosier = self.spotasocNoiser()

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=self.collate_fn, batch_size=self.args.train_batchsize, num_workers=self.args.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=self.collate_fn, batch_size=self.args.val_batchsize,num_workers=self.args.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, collate_fn=self.collate_fn, batch_size=self.args.val_batchsize, num_workers=self.args.num_workers, pin_memory=False)

    def predict_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, collate_fn=self.collate_fn, batch_size=self.args.val_batchsize, num_workers=self.args.num_workers, pin_memory=False)

    def spotasocNoiser(self):
        if self.args.source_prefix.startswith('meta'):
            if self.args.spot_noise > 0 or self.args.asoc_noise > 0:
                if self.args.decoding_format == 'spotasoc':
                    spot_asoc_nosier = SpotAsocNoiser(
                        spot_noise_ratio=self.args.spot_noise,
                        asoc_noise_ratio=self.args.asoc_noise,
                        null_span=constants.null_span,
                    )
                else:
                    raise NotImplementedError( "decoding_format `spotasoc` is not implemented." )
                    
                return spot_asoc_nosier
        return None
        
    def dynamicSSIGenerator(self):
        return DynamicSSIGenerator(tokenizer=self.tokenizer,
                                    schema=self.schema,
                                    positive_rate=self.args.meta_positive_rate, 
                                    negative=self.args.negative ,
                                    ordered_prompt =self.args.ordered_prompt)
        
    def collate_fn(self, features):
        for feature in features:
            sample_prompt = feature['sample_prompt']
            if not sample_prompt:
                # Evaluation using Ordered SSI
                converted_spot_prefix = self.negative_sampler.full_spot(shuffle=False) # To FIX
                converted_asoc_prefix = self.negative_sampler.full_asoc(shuffle=False)
            else:
                # Sample SSI
                converted_spot_prefix, positive_spot, negative_spot = self.negative_sampler.sample_spot(positive=feature.get('spots', []))
                converted_asoc_prefix, negative_asoc = self.negative_sampler.sample_asoc(positive=feature.get('asocs', []))

                # Dynamic generating spot-asoc during training
                if 'spot_asoc' in feature:
                    # Deleted positive example Spot in Target that was not sampled by Prefix
                    feature['spot_asoc'] = [spot_asoc for spot_asoc in feature['spot_asoc'] if spot_asoc["label"] in positive_spot]
                    # Inject rejection noise
                    if self.spot_asoc_nosier is not None:
                        if isinstance(self.spot_asoc_nosier, SpotAsocNoiser):
                            feature['spot_asoc'] = self.spot_asoc_nosier.add_noise(
                                feature['spot_asoc'],
                                spot_label_list=negative_spot,
                                asoc_label_list=negative_asoc,
                            )
                        else:
                            raise NotImplementedError(f'{self.spot_asoc_nosier} is not implemented.')
                    # Generate new record
                    record = convert_to_record_function[self.args.decoding_format](
                        feature['spot_asoc'],
                        structure_maker=BaseStructureMarker()
                    )
                    feature["labels"] = self.tokenizer.encode(record, add_special_tokens=False)

            feature.pop('sample_prompt') if 'sample_prompt' in feature else None # if sample
            feature.pop('spot_asoc') if 'spot_asoc' in feature else None
            feature.pop('spots') if 'spots' in feature else None
            feature.pop('asocs') if 'asocs' in feature else None
            
            prefix = converted_spot_prefix + converted_asoc_prefix
 
 
            if self.args.max_prefix_length is not None and self.args.max_prefix_length >= 0:  
                prefix = prefix[:self.args.max_prefix_length] 
            feature['input_ids'] = prefix + [self.negative_sampler.text_start] + feature['input_ids'] 

            if self.args.max_length:
                feature['input_ids'] = feature['input_ids'][:self.args.max_length]
            if self.args.max_target_length and 'labels' in feature:
                feature['labels'] = feature['labels'][:self.args.max_target_length]
            feature['attention_mask'] = [1] * len(feature['input_ids'])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(_label) for _label in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.args.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
        features = self.tokenizer.pad(features,
            padding=self.args.padding,
            max_length=self.args.max_length,
            pad_to_multiple_of=None,  #TODO 如果使用半精度训练的时候设置为8才发现
            return_tensors="pt"
        )

        # TODO 只有在配合使用平滑（label smoothing）的时候才用到, 防止溢出
        # prepare decoder_input_ids 
        # if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        #     features["decoder_input_ids"] = decoder_input_ids

        return features

        
class UIEModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--predict_with_generate', default=True, type=bool )
        parser.add_argument('--predict_loss_only', default=True, type=bool)
        parser.add_argument('--num_beams', default=1, type=int)
        
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--check_val_every_n_epoch',default=2, type=int)
        
        parser.add_argument('--gpus', default='0', type=str, help='0')    
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        
        parser.add_argument('--warmup_steps', default=0, type=int, help='') # TODO 
        parser.add_argument('--warmup_ratio', default=0.06, type=float)
        parser.add_argument('--adam_beta1', default=0.9, type=float)
        parser.add_argument('--adam_beta2', default=0.999, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--scheduler_type', default='linear', type=str, choices=['linear', 'polynomial', 'cosine', 'constant'])
        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_path)
        self.tokenizer =tokenizer

    def add_special_tokens(self):
        to_add_special_token = list()
        for special_token in [constants.type_start, constants.type_end, constants.text_start, constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
            if special_token not in self.tokenizer.get_vocab():
                to_add_special_token += [special_token]
        tokenizer_sp = self.tokenizer.special_tokens_map_extended['additional_special_tokens'] if 'additional_special_tokens' in self.tokenizer.special_tokens_map_extended else []
        self.tokenizer.add_special_tokens({"additional_special_tokens": tokenizer_sp + to_add_special_token})
        self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info(self.tokenizer)
           
    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader() #
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)  
                self.total_steps = (len(train_loader.dataset)*self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            logger.info('Total steps: {}' .format(self.total_steps))
            logger.info('Estimated  steps: {}'.format(self.trainer.estimated_stepping_batches))

  
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay': self.args.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=self.args.learning_rate)
        warmup_steps = self.args.warmup_ratio*self.total_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                num_warmup_steps=int(warmup_steps), num_training_steps=self.total_steps)
        return [optimizer], [scheduler]
            
    def training_step(self, batch, batch_idx):        
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        
        self.log('train_loss', output.loss, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        generated_tokens = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            num_beams=self.args.num_beams,
        )
        results = self.comput_metrix(generated_tokens, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log_dict(results)
        
    def predict_step(self, batch, batch_idx):

        generated_tokens = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            num_beams=self.args.num_beams,
        )
        
        preds = self.tokenizer.batch_decode(generated_tokens,
                                            skip_special_tokens=False, 
                                            clean_up_tokenization_spaces=False)
        preds = [self.postprocess_text(x) for x in preds]
        
        return preds
        
    def postprocess_text(self, x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        to_remove_token_list = list()
        if self.tokenizer.bos_token:
            to_remove_token_list += [self.tokenizer.bos_token]
        if self.tokenizer.eos_token:
            to_remove_token_list += [self.tokenizer.eos_token]
        if self.tokenizer.pad_token:
            to_remove_token_list += [self.tokenizer.pad_token]
        for to_remove_token in to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()
    
    def comput_metrix(self, preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if self.args.ignore_pad_token_for_loss:
            labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [self.postprocess_text(x) for x in decoded_preds]
        decoded_labels = [self.postprocess_text(x) for x in decoded_labels]

        result = get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=RecordSchema.read_from_file(self.args.record_schema),
            decoding_format=self.args.decoding_format,
        )
        prediction_lens = [torch.count_nonzero(pred != self.tokenizer.pad_token_id).tolist() for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


class UIEModelCallbacks:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('Base Callbacks')
        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./log/', type=str)
        parser.add_argument('--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)

        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--log_every_n_step', default=100, type=float)
        parser.add_argument('--logging_interval', default='step', type=str)
        
        parser.add_argument('--every_n_epochs', default=1, type=float)
        parser.add_argument('--every_n_train_steps', default=100, type=float)
        parser.add_argument('--save_weights_only', default=True, type=bool)
        return parent_args

    def __init__(self, args):
        self.checkpoint_callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         save_last=True,
                                         every_n_epochs=args.every_n_epochs,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.dirpath,
                                         filename=args.filename)
        
        self.lr_callbacks = LearningRateMonitor(logging_interval='step')
  
     
def add_special_tokens(tokenizer):
    to_add_special_token = list()
    for special_token in [constants.type_start, constants.type_end, constants.text_start, constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
        if special_token not in tokenizer.get_vocab():
            to_add_special_token += [special_token]
    tokenizer_sp = tokenizer.special_tokens_map_extended.get('additional_special_tokens',[]) 
    tokenizer.add_special_tokens({"additional_special_tokens": tokenizer_sp + to_add_special_token})
    logger.info(tokenizer)
    return tokenizer
    
def main():
    total_parser = argparse.ArgumentParser("UIE ")
    
    total_parser.add_argument('--pretrained_model_path', default='', type=str)
    total_parser.add_argument('--checkpoint_path', default='' ,type=str)
    total_parser.add_argument('--tokenizer_type', default='t5bert', type=str, choices=['t5bert', 't5sp'])
    total_parser.add_argument('--train_file', default='train.json', type=str)
    total_parser.add_argument('--valid_file', default='test.json', type=str)
    total_parser.add_argument('--test_file', default='test.json', type=str)
    total_parser.add_argument('--record_schema', default='record_schema', type=str)

    total_parser.add_argument('--do_train',action='store_true')
    
    total_parser.add_argument('--precision', default=32, type=int)
    total_parser.add_argument('--seed', default=42, type=int)
    # total_parser.add_argument('--strategy',default='ddp', type=str)
    

    total_parser = UIEDataset.add_data_specific_args(total_parser)
    total_parser = UIEDataModel.add_data_specific_args(total_parser)
    total_parser = UIEModel.add_model_specific_args(total_parser)
    total_parser = UIEModelCallbacks.add_argparse_args(total_parser)
    args = total_parser.parse_args()
    seed_everything(args.seed)
  
  
    if args.tokenizer_type == 't5bert':
        tokenizer = T5BertTokenizer.from_pretrained(args.pretrained_model_path)
        tokenizer =add_special_tokens(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        tokenizer = add_special_tokens(tokenizer)

    
    datamodel= UIEDataModel(train_file=args.train_file, 
                            valid_file=args.valid_file, 
                            test_file=args.test_file,
                            tokenizer=tokenizer, args=args)

    model = UIEModel(args, tokenizer)
    callbacks = [UIEModelCallbacks(args).checkpoint_callbacks, 
                 UIEModelCallbacks(args).lr_callbacks]
    
    logger = loggers.TensorBoardLogger(save_dir=args.dirpath)
    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=callbacks)
    if args.do_train:
        trainer.fit(model, datamodel)
        
    if args.checkpoint_path!='':
        model =UIEModel.load_from_checkpoint(args.checkpoint_path, args=args, tokenizer=tokenizer)
        model =model.cuda()
    else:
        checkpoint_path = os.path.join(args.dirpath, 'last.ckpt')
        assert checkpoint_path != ''
        print("Load from {} Sucessfully!".format(checkpoint_path))
        model =UIEModel.load_from_checkpoint( checkpoint_path=checkpoint_path,
                                            args=args, tokenizer=tokenizer)
        model =model.cuda()
    trainer.validate(model, datamodel)


if __name__ == '__main__':
    main()

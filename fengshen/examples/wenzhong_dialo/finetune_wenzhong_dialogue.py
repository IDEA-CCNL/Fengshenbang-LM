from random import sample
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_lightning import (
    LightningModule,
    Trainer,
    loggers,
)
from torchtext.data.metrics import bleu_score
from pytorch_lightning.callbacks import LearningRateMonitor

import os
import argparse
import torch, logging, traceback
import sys
sys.path.append('../../')
sys.path.append('/home/yangqi/Fengshenbang-LM/')
from fengshen.data.gpt_dataloader import DusincDataModule, MixingDataModule, DialoCollator, QueryCollator
from fengshen.utils import UniversalCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


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
        # get loss
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'])

        acc = self.comput_metrix(output.logits, batch["labels"])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)

        #get output
        # generated_ids = self.model.generate(
        #     input_ids=batch['input_ids'],
        #     attention_mask=batch['attention_mask']
        #     )
        
        # preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # kno = batch['knowledge']
        # qus = batch['question']
        # file_path = os.path.join(self.trainer.checkpoint_callback.dirpath, f"{self.hparams.resume_from_checkpoint}.json")
        # self.save_prediction_to_file(preds, labels, kno, qus, file_path)
        # self.log('val_rouge', output.loss, sync_dist=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

    def save_prediction_to_file(self, preds, labels, knos, ques, file_path):
        import json
        with open(file_path, 'a', encoding='utf-8') as f:
            for idx, pred in enumerate(preds):
                label,kno,qus = labels[idx], knos[idx], ques[idx]
                tmp_result = dict()
                tmp_result['pred'] = pred
                tmp_result['label'] = label
                tmp_result['kno'] = kno
                tmp_result['qus'] = qus
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data + '\n')

    def predict_step(self, batch, batch_idx):
        # print(batch)
        kno = batch['knowledge']
        qus = batch["question"]
        # output summary and metrics
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        labels = self.tokenizer.batch_decode(
            batch['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(batch_idx, len(preds), len(labels))
        file_path = os.path.join(self.trainer.checkpoint_callback.dirpath, f"{self.hparams.resume_from_checkpoint}.json")
        self.save_prediction_to_file(preds, labels, kno, qus, file_path)

if __name__ == '__main__':

    
    try: 
        args_parser = argparse.ArgumentParser()
        args_parser = DusincDataModule.add_data_specific_args(args_parser)
        args_parser = Trainer.add_argparse_args(args_parser)
        args_parser = GPT2Finetuner.add_module_specific_args(args_parser)
        args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
        args_parser = DialoCollator.add_data_specific_args(args_parser)
        args_parser.add_argument('--deepspeed')
        args_parser.add_argument('--do_eval_only', action='store_true', default=False)
        args_parser.add_argument('--pretrain_sp_tokenizer', type=str,default='')
        args_parser.add_argument('--task', type=str,default='dial')
        args_parser.add_argument('--device', type=str,default='cuda:5')
        args = args_parser.parse_args()

        # ! debug code
        logging.basicConfig(filename=os.path.join(args.default_root_dir,'traceback2.log'),
            level=logging.INFO, filemode='a', 
            format='[%(asctime)s] [%(levelname)s] >>>  %(message)s',
            datefmt='%Y-%m-%d %I:%M:%S')
        logging.debug(args)

        tokenizer = GPT2Tokenizer.from_pretrained(args.pretrain_sp_tokenizer,extra_ids=0)
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'}) #[PAD]

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


        if not args.do_eval_only:
            logging.info("Begin training")
            trainer = Trainer.from_argparse_args(args, logger=logger,
                                                callbacks=[
                                                lr_monitor,
                                                checkpoint_callback])
            trainer.fit(module, data_module)
        else:
            trainer = Trainer.from_argparse_args(args)
            trainer.validate(module, data_module)
        
        #trainer.test(ckpt_path=trainer.checkpoint_callback.last_model_path,             dataloaders=data_module, verbose=True)
    except Exception as e:
        logging.error("Main program error:")
        logging.error(e)
        logging.error(traceback.format_exc())   
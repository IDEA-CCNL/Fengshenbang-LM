import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from dataclasses import dataclass
from typing import Dict, List, Union

from fengshen.models.tagging_models.bert_for_tagging import BertLinear,BertCrf,BertSpan,BertBiaffine
from fengshen.data.tag_dataloader.tag_collator import CollatorForLinear, CollatorForCrf, CollatorForSpan, CollatorForBiaffine
from fengshen.data.tag_dataloader.tag_datamodule import TaskDataModel
from fengshen.data.tag_dataloader.tag_datasets import get_labels
from fengshen.metric.metric import EntityScore
from fengshen.metric.utils_ner import get_entities, bert_extract_item
from transformers import (
    BertConfig,
    AutoTokenizer, BertTokenizer
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.pipelines.base import PipelineException, GenericTensor
from transformers import TokenClassificationPipeline as HuggingfacePipe
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.models.model_utils import add_module_args
from fengshen.models.model_utils import configure_optimizers

_model_dict={
    'bert-linear': BertLinear,
    'bert-crf': BertCrf,
    'bert-span': BertSpan,
    'bert-biaffine': BertBiaffine
}

_collator_dict={
    'linear': CollatorForLinear,
    'crf': CollatorForCrf,
    'span': CollatorForSpan,
    'biaffine': CollatorForBiaffine
}


class _taskModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('sequence tagging task model')
        parser.add_argument('--model_type', default='bert', type=str)
        parser.add_argument('--loss_type', default='ce', type=str)
        return parent_args
    
    def __init__(self, args, model, label2id, valiate_fn):
        super().__init__()
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.model=model
        self.validate_fn=getattr(self,valiate_fn)

        self.entity_score=EntityScore()

        self.save_hyperparameters(args)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches
                self.total_steps = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.validate_fn(batch,batch_idx)

    def validation_linear(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        preds = preds.detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()

        for i, label in enumerate(labels):
            y_true = []
            y_pred = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == (torch.sum(batch['attention_mask'][i]).item()-1):
                    true_subject=get_entities(y_true,self.id2label)
                    pred_subject=get_entities(y_pred,self.id2label)
                    self.entity_score.update(true_subject=true_subject, pred_subject=pred_subject)
                    break
                else:
                    y_true.append(self.id2label[labels[i][j]])
                    y_pred.append(self.id2label[preds[i][j]])
        
        self.log('val_loss', loss)

    def validation_crf(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = self.model.crf.decode(logits, batch['attention_mask'])
        preds = preds.detach().squeeze(0).cpu().numpy().tolist()
        labels = batch['labels'].detach().cpu().numpy()

        for i, label in enumerate(labels):
            y_true = []
            y_pred = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == (torch.sum(batch['attention_mask'][i]).item()-1):
                    true_subject=get_entities(y_true,self.id2label)
                    pred_subject=get_entities(y_pred,self.id2label)
                    self.entity_score.update(true_subject=true_subject, pred_subject=pred_subject)
                    break
                else:
                    y_true.append(self.id2label[labels[i][j]])
                    y_pred.append(self.id2label[preds[i][j]])

        self.log('val_loss', loss)

    def validation_span(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        labels=batch['subjects']
        for i, T in enumerate(labels):
            active_start_logits=start_logits[i][:batch['input_len'][i]]
            active_end_logits=end_logits[i][:batch['input_len'][i]]
            R = bert_extract_item(active_start_logits, active_end_logits)

            T=T[~torch.all(T==-1,dim=-1)].cpu().numpy()
            T=list(map(lambda x:(self.id2label[x[0]],x[1],x[2]),T))
            R=list(map(lambda x:(self.id2label[x[0]],x[1],x[2]),R))

            self.entity_score.update(true_subject=T, pred_subject=R)
        self.log('val_loss', loss)

    def validation_biaffine(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.span_logits

        preds = torch.argmax(logits.cpu().numpy(), axis=-1)
        labels = batch['span_labels'].cpu().numpy()

        for i, label in enumerate(labels):
            input_len=(batch['input_len'][i])-2
            active_label=labels[i,1:input_len+1,1:input_len+1]
            active_pred=preds[i,1:input_len+1,1:input_len+1]

            temp_1 = []
            temp_2 = []

            for j in range(input_len):
                for k in range(input_len):
                    if self.id2label[active_label[j,k]]!="O":
                        temp_1.append([self.id2label[active_label[j,k]],j,k])
                    if self.id2label[active_pred[j,k]]!="O":
                        temp_2.append([self.id2label[active_pred[j,k]],j,k])

            self.entity_score.update(pred_subject=temp_2, true_subject=temp_1)

        self.log('val_loss', loss)
    
    def validation_epoch_end(self, outputs):
        # compute metric for all process
        score_dict, _ = self.entity_score.result()
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            print('score_dict:\n', score_dict)
        # reset the metric after once validation
        self.entity_score.reset()
        for k, v in score_dict.items():
            self.log('val_{}'.format(k), v)

    def configure_optimizers(self):
        return configure_optimizers(self)


class SequenceTaggingPipeline(HuggingfacePipe):
    @staticmethod
    def add_pipeline_specific_args(parent_args):
        parser = parent_args.add_argument_group('SequenceTaggingPipeline')
        parser = _taskModel.add_model_specific_args(parent_args)
        parser = TaskDataModel.add_data_specific_args(parent_args)
        parser = UniversalCheckpoint.add_argparse_args(parent_args)
        parser = pl.Trainer.add_argparse_args(parent_args)
        parser = add_module_args(parent_args)
        return parent_args

    def __init__(self,
                model_path: str = None,
                args=None,
                **kwargs):

        _validation_dict={
            'linear': 'validation_linear',
            'crf': 'validation_crf',
            'span': 'validation_span',
            'biaffine': 'validation_biaffine',
        }

        _prediction_dict={
            'linear': 'postprocess_linear',
            'crf': 'postprocess_crf',
            'span': 'postprocess_span',
            'biaffine': 'postprocess_biaffine',
        }

        self.args = args
        self.model_name=args.model_type+"-"+args.decode_type

        self.label2id,self.id2label=get_labels(args.decode_type)

        self.config=BertConfig.from_pretrained(model_path)
        self.model = _model_dict[self.model_name].from_pretrained(model_path, config=self.config, num_labels=len(self.label2id), loss_type=args.loss_type)
        self.tokenizer=BertTokenizer.from_pretrained(model_path)
        self.validate_fn = _validation_dict[args.decode_type]
        self.predict_fn = getattr(self,_prediction_dict[args.decode_type])

        self.collator = _collator_dict[args.decode_type]()
        self.collator.args=self.args
        self.collator.tokenizer=self.tokenizer
        self.collator.label2id=self.label2id

        device=-1
        super().__init__(model=self.model,
                         tokenizer=self.tokenizer,
                         framework='pt',
                         device=device,
                         **kwargs)

    def check_model_type(self, supported_models: Union[List[str], dict]):
        pass

    def train(self):
        checkpoint_callback = UniversalCheckpoint(self.args).callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer.from_argparse_args(self.args,
                                            callbacks=[checkpoint_callback, lr_monitor]
                                            )

        data_model = TaskDataModel(args=self.args,collate_fn=self.collator,tokenizer=self.tokenizer)
        model = _taskModel(self.args,self.model,self.label2id,self.validate_fn)

        trainer.fit(model,data_model)

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return (model_inputs,outputs)

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        samples=[]
        labels,subject=["O" for _ in range(len(inputs))],[]
        samples.append({"text_a": list(inputs), "labels": labels, "subject":subject})
        return self.collator(samples)

    def postprocess(self, model_outputs):
        return self.predict_fn(model_outputs)

    def postprocess_linear(self, model_outputs):
        model_inputs,outputs=model_outputs
        preds = torch.argmax(F.log_softmax(outputs.logits, dim=2), dim=2)
        preds = preds.detach().cpu().numpy()
        text = self.tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])[:model_inputs['input_len'][0]][1:-1]
        pred = preds[0][:model_inputs['input_len'][0]][1:-1]
        label_entities = get_entities(pred, self.id2label)
        for label_list in label_entities:
            label_list.append("".join(text[label_list[1]:label_list[2]+1]))

        return label_entities
    
    def postprocess_crf(self, model_outputs):
        model_inputs,outputs=model_outputs
        preds = self.model.crf.decode(outputs.logits, model_inputs['attention_mask']).squeeze(0).cpu().numpy().tolist()
        text = self.tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])[:model_inputs['input_len'][0]][1:-1]
        pred = preds[0][:model_inputs['input_len'][0]][1:-1]
        label_entities = get_entities(pred, self.id2label)
        for label_list in label_entities:
            label_list.append("".join(text[label_list[1]:label_list[2]+1]))

        return label_entities
    
    def postprocess_span(self, model_outputs):
        model_inputs,outputs=model_outputs

        start_logits, end_logits = outputs.start_logits[0], outputs.end_logits[0]
        text = self.tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])[:model_inputs['input_len'][0]][1:-1]
        R = bert_extract_item(start_logits[:model_inputs['input_len'][0]], end_logits[:model_inputs['input_len'][0]])
        label_entities = [[self.id2label[x[0]],x[1],x[2],"".join(text[x[1]:x[2]+1])] for x in R]

        return label_entities


Pipeline = SequenceTaggingPipeline

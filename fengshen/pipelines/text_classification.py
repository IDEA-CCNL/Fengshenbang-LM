import torch
from torch.utils.data._utils.collate import default_collate
from dataclasses import dataclass
from typing import Dict, List
from .base import (
    _CONFIG_MODEL_TYPE,
    _CONFIG_TOKENIZER_TYPE)
from fengshen.models.roformer import RoFormerForSequenceClassification
from fengshen.models.longformer import LongformerForSequenceClassification
from fengshen.models.zen1 import ZenForSequenceClassification
from transformers import (
    BertConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.pipelines.base import PipelineException, GenericTensor
from transformers import TextClassificationPipeline as HuggingfacePipe
import pytorch_lightning as pl
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.models.model_utils import add_module_args
import torchmetrics

_model_dict = {
    'fengshen-roformer': RoFormerForSequenceClassification,
    # 'fengshen-megatron_t5': T5EncoderModel,  TODO 实现T5EncoderForSequenceClassification
    'fengshen-longformer': LongformerForSequenceClassification,
    'fengshen-zen1': ZenForSequenceClassification,
    'huggingface-auto': AutoModelForSequenceClassification,
}

_tokenizer_dict = {}

_ATTR_PREPARE_INPUT = '_prepare_inputs_for_sequence_classification'


class _taskModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        _ = parent_args.add_argument_group('text classification task model')
        return parent_args

    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.acc_metrics = torchmetrics.Accuracy()
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
        loss, _ = outputs[0], outputs[1]
        self.log('train_loss', loss)
        return loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).long()
        acc = self.acc_metrics(y_pred.long(), y_true.long())
        return acc

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss, logits = outputs[0], outputs[1]
        acc = self.comput_metrix(logits, batch['labels'])
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def predict_step(self, batch, batch_idx):
        output = self.model(**batch)
        return output.logits

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)


@dataclass
class _Collator:
    tokenizer = None
    texta_name = 'sentence'
    textb_name = 'sentence2'
    label_name = 'label'
    max_length = 512
    model_type = 'huggingface-auto'

    def __call__(self, samples):
        sample_list = []
        for item in samples:
            if self.textb_name in item and item[self.textb_name] != '':
                if self.model_type != 'fengshen-roformer':
                    encode_dict = self.tokenizer.encode_plus(
                        [item[self.texta_name], item[self.textb_name]],
                        max_length=self.max_length,
                        padding='max_length',
                        truncation='longest_first')
                else:
                    encode_dict = self.tokenizer.encode_plus(
                        [item[self.texta_name]+'[SEP]'+item[self.textb_name]],
                        max_length=self.max_length,
                        padding='max_length',
                        truncation='longest_first')
            else:
                encode_dict = self.tokenizer.encode_plus(
                    item[self.texta_name],
                    max_length=self.max_length,
                    padding='max_length',
                    truncation='longest_first')
            sample = {}
            for k, v in encode_dict.items():
                sample[k] = torch.tensor(v)
            if self.label_name in item:
                sample['labels'] = torch.tensor(item[self.label_name]).long()
            sample_list.append(sample)
        return default_collate(sample_list)


class TextClassificationPipeline(HuggingfacePipe):
    @staticmethod
    def add_pipeline_specific_args(parent_args):
        parser = parent_args.add_argument_group('SequenceClassificationPipeline')
        parser.add_argument('--texta_name', default='sentence', type=str)
        parser.add_argument('--textb_name', default='sentence2', type=str)
        parser.add_argument('--label_name', default='label', type=str)
        parser.add_argument('--max_length', default=512, type=int)
        parser.add_argument('--device', default=-1, type=int)
        parser = _taskModel.add_model_specific_args(parent_args)
        parser = UniversalDataModule.add_data_specific_args(parent_args)
        parser = UniversalCheckpoint.add_argparse_args(parent_args)
        parser = pl.Trainer.add_argparse_args(parent_args)
        parser = add_module_args(parent_args)
        return parent_args

    def __init__(self,
                 model: str = None,
                 args=None,
                 **kwargs):
        self.args = args
        self.model_name = model
        self.model_type = 'huggingface-auto'
        # 用BertConfig做兼容，我只需要读里面的fengshen_model_type，所以这里用啥Config都可以
        config = BertConfig.from_pretrained(model)
        if hasattr(config, _CONFIG_MODEL_TYPE):
            self.model_type = config.fengshen_model_type
        if self.model_type not in _model_dict:
            raise PipelineException(self.model_name, ' not in model type dict')
        # 加载模型，并且使用模型的config
        self.model = _model_dict[self.model_type].from_pretrained(model)
        self.config = self.model.config
        # 加载分词
        tokenizer_config = get_tokenizer_config(model, **kwargs)
        self.tokenizer = None
        if hasattr(tokenizer_config, _CONFIG_TOKENIZER_TYPE):
            if tokenizer_config._CONFIG_TOKENIZER_TYPE in _tokenizer_dict:
                self.tokenizer = _tokenizer_dict[tokenizer_config._CONFIG_TOKENIZER_TYPE].from_pretrained(
                    model)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        # 加载数据处理模块
        c = _Collator()
        c.tokenizer = self.tokenizer
        c.model_type = self.model_type
        if args is not None:
            c.texta_name = self.args.texta_name
            c.textb_name = self.args.textb_name
            c.label_name = self.args.label_name
            c.max_length = self.args.max_length
        self.collator = c
        device = -1 if args is None else args.device
        super().__init__(model=self.model,
                         tokenizer=self.tokenizer,
                         framework='pt',
                         device=device,
                         **kwargs)

    def train(self,
              datasets: Dict):
        """
        Args:
            datasets is a dict like
            {
                test: Dataset()
                validation: Dataset()
                train: Dataset()
            }
        """
        checkpoint_callback = UniversalCheckpoint(self.args)
        trainer = pl.Trainer.from_argparse_args(self.args,
                                                callbacks=[checkpoint_callback]
                                                )

        data_model = UniversalDataModule(
            datasets=datasets,
            tokenizer=self.tokenizer,
            collate_fn=self.collator,
            args=self.args)
        model = _taskModel(self.args, self.model)

        trainer.fit(model, data_model)
        return

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        # 如果模型有自定义的接口，用模型的口
        if hasattr(self.model, _ATTR_PREPARE_INPUT):
            return getattr(self.model, _ATTR_PREPARE_INPUT)(inputs, self.tokenizer, **tokenizer_kwargs)
        samples = []
        if isinstance(inputs, str):
            samples.append({self.collator.texta_name: inputs})
        else:
            # 在__call__里面已经保证了input的类型，所以这里直接else就行
            for i in inputs:
                samples.append({self.collator.texta_name})
        return self.collator(samples)


Pipeline = TextClassificationPipeline

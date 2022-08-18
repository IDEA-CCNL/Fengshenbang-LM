from transformers import (
    AutoProcessor,
)
from transformers import Wav2Vec2Config
from fengshen.models.wav2vec2.modeling_wav2vec import Wav2Vec2ForCTC
from datasets import load_metric
import fengshen.data.wav2vec2.wav2vec2ctc_dataset as ctc_datasets
from fengshen.data.universal_datamodule import UniversalDataModule
# from transformers.models.hubert.modeling_hubert import _compute_mask_indices
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
    loggers,
)
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import os


class Wav2vec2CTCDataLoader():
    def __init__(self, args, model):
        self.args = args
        self.load_datasets = {}
        self.wav2vec2_model = model.model
        self.feature_extractor = model.feature_extractor
        self.tokenizer = model.tokenizer
        self.processor = model.processor

    @property
    def datasets(self):
        return self.load_datasets

    def load_dataset(self, split: str, **kwargs):
        data_path = self.args.data
        manifest_path = os.path.join(data_path, "{}.tsv".format(split))
        label_path = os.path.join(data_path, "{}.wrd".format(split))

        self.datasets[split] = ctc_datasets.CTCDataset(
            manifest_path=manifest_path,
            lable_path=label_path,
            sample_rate=self.args.sample_rate,
            processor=self.processor,
            args=self.args,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.min_sample_size,
            pad=self.args.labels is not None or self.args.enable_padding,
            normalize=self.args.normalize,
        )


def prepare_data(args, model):
    loader = Wav2vec2CTCDataLoader(args, model)
    loader.load_dataset('train')
    loader.load_dataset('valid')
    return loader


class CTCLightning(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Wav2vec2 Lightning')
        parser.add_argument(
            "--audio_column_name",
            type=str,
            default="audio",
            help="Column in the dataset that contains speech file path. Defaults to 'audio'",
        )
        parser.add_argument(
            "--max_gumbel_temperature",
            type=float,
            default=2.0,
            help="Maximum temperature for gumbel softmax.",
        )
        parser.add_argument(
            "--min_gumbel_temperature",
            type=float,
            default=0.5,
            help="Minimum temperature for gumbel softmax.",
        )
        parser.add_argument(
            "--gumbel_temperature_decay", type=float, default=0.999995,
            help="Decay of gumbel temperature during training."
        )
        parser.add_argument(
            "--pad_to_multiple_of",
            type=int,
            default=None,
            help=(
                "If set will pad the sequence to a multiple of the provided value. "
                "This is especially useful to enable the"
                " use of Tensor Cores on NVIDIA hardware with compute capability"
                " >= 7.5 (Volta)."
            ),
        )
        parser.add_argument(
            "--feature_grad_mult",
            type=float,
            default=1,
            help="modified the feature encoder's gradient"
        )

        parser.add_argument(
            "--pretrained_model",
            type=str,
            default=None
        )

        parser.add_argument("--unk_token", default=None)
        parser.add_argument("--pad_token", default=None)
        parser.add_argument("--word_delimiter_token", default=None)
        parser.add_argument("--eval_metrics", default=["wer"], nargs='*')

        return parent_parser

    def __init__(self, args, ** kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        config = Wav2Vec2Config.from_pretrained(args.model_path)
        self.config = config

        # tokenizer_kwargs = {
        #     "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
        # }
        tokenizer_kwargs = dict()
        if args.unk_token is not None:
            tokenizer_kwargs["unk_token"] = args.unk_token
        if args.pad_token is not None:
            tokenizer_kwargs["pad_token"] = args.pad_token
        if args.word_delimiter_token is not None:
            tokenizer_kwargs["word_delimiter_token"] = args.word_delimiter_token
        self.processor = AutoProcessor.from_pretrained(args.model_path, **tokenizer_kwargs)
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer
        config.update(
            {
                "pad_token_id": self.tokenizer.pad_token_id,
                "vocab_size": len(self.tokenizer),
            }
        )
        config.feature_grad_mult = args.feature_grad_mult
        config.vocab_size = len(self.processor.tokenizer.get_vocab())
        if args.pretrained_model:
            self.model = Wav2Vec2ForCTC.from_pretrained(args.pretrained_model, config=config)
        else:
            self.model = Wav2Vec2ForCTC(config=config)

        # self.model.freeze_feature_encoder()
        # used to update gumbel temperature

        # print(name)

        self.eval_metrics = {metric: load_metric(metric) for metric in args.eval_metrics}
        # used for accumulate gradient

    def setup(self, stage) -> None:
        if stage == 'fit':
            # for name, item in self.named_parameters():
            #     name_split = name.split(".")
            #     if name_split[0]!="model" or name_split[1]!="lm_head":
            #         item.requires_grad = False
            if self.args.pretrained_model:
                self.model.freeze_feature_encoder()
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

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)

    def forward(self, **batch):
        batch.pop("sub_attention_mask", None)
        output = self.model(
            **batch
        )
        return output

    def compute_metrics(self, batch, pred):
        pred_logits = pred.logits
        pred_ids = torch.argmax(pred_logits, axis=-1)
        label_ids = batch["labels"].clone()
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.tokenizer.batch_decode(label_ids, group_tokens=False)
        # pred_str = [" ".join(item) for item in pred_str]
        # label_str = [" ".join(item) for item in label_str]

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in self.eval_metrics.items()}

        return metrics

    def multiply_grads(self, params, c):
        """Multiplies grads by a constant *c*."""
        for p in params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        logs = {
            "train_loss": output.loss,
        }
        self.log("train", logs)
        if self.trainer.accumulate_grad_batches:
            return {
                "loss": output.loss/self.trainer.accumulate_grad_batches,
            }
        else:
            return {
                "loss": output.loss,
            }

    def validation_step(self, batch, batch_idx):
        # print([item for item in batch])
        output = self(**batch)
        metrics = self.compute_metrics(batch, output)
        self.log("evaluate", metrics)
        self.log("valid", {
            "valid_loss": output.loss,
        })
        return {
            "loss": output.loss,
        }

    def on_save_checkpoint(self, checkpoint) -> None:
        # Save the current loop info in the mid of epoch
        # if you lightning <= 1.6.0  uncomment the line below
        # checkpoint['loops'] = self.trainer.checkpoint_connector._get_loops_state_dict()
        if self.trainer.global_rank == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(self.trainer.current_epoch, self.trainer.global_step)))

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    from fengshen.utils import UniversalCheckpoint
    from fengshen.models.model_utils import add_module_args
    args_parser = add_module_args(args_parser)
    args_parser = ctc_datasets.add_data_specific_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = CTCLightning.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser.add_argument('--ckpt_path', type=str, )
    args = args_parser.parse_args()

    module = CTCLightning(args)
    data_loader = prepare_data(args, module)
    data_module = UniversalDataModule(
        args=args, tokenizer=None, datasets=data_loader.datasets, collate_fn=None)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        args.default_root_dir, 'logs/'),
        name=os.path.basename(os.path.dirname(args.model_path)))
    checkpoint_callback = UniversalCheckpoint(args).callbacks

    if args.ckpt_path is not None and \
            not os.path.exists(args.ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.ckpt_path = None

    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(module, data_module, ckpt_path=args.ckpt_path)

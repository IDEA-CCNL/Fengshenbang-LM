from pytorch_lightning import LightningModule

from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers.optimization import AdamW, get_scheduler


def add_module_args(parent_args):
    parser = parent_args.add_argument_group('Basic Module')
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--adam_beta1', default=0.9, type=float)
    parser.add_argument('--adam_beta2', default=0.999, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--scheduler_type', default='polynomial', type=str)
    return parent_args


def configure_optimizers(pl_model: LightningModule):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_params = [
        {'params': [p for n, p in pl_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': pl_model.hparams.weight_decay},
        {'params': [p for n, p in pl_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # Configure optimizer.
    if isinstance(pl_model.trainer.strategy, DeepSpeedStrategy):
        if 'offload_optimizer' in pl_model.trainer.training_type_plugin.config['zero_optimization']:
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_params, adamw_mode=True,
                lr=pl_model.hparams.learning_rate,
                betas=(pl_model.hparams.adam_beta1, pl_model.hparams.adam_beta2), eps=pl_model.hparams.adam_epsilon)
        else:
            optimizer = FusedAdam(
                optimizer_grouped_params, adam_w_mode=True,
                lr=pl_model.hparams.learning_rate,
                betas=(pl_model.hparams.adam_beta1, pl_model.hparams.adam_beta2), eps=pl_model.hparams.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_params, lr=pl_model.hparams.learning_rate,
                          betas=(pl_model.hparams.adam_beta1, pl_model.hparams.adam_beta2),
                          eps=pl_model.hparams.adam_epsilon)
    # Configure learning rate scheduler.
    warmup_steps = pl_model.hparams.warmup_ratio * \
        pl_model.total_steps if pl_model.hparams.warmup_steps == 0 else pl_model.hparams.warmup_steps
    scheduler = get_scheduler(name=pl_model.hparams.scheduler_type, optimizer=optimizer,
                              num_warmup_steps=warmup_steps, num_training_steps=pl_model.total_steps)
    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return [optimizer], [scheduler]

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers.optimization import AdamW, TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_utils import SchedulerType
from typing import Optional, Union
from torch.optim import Optimizer


def add_module_args(parent_args):
    parser = parent_args.add_argument_group('Basic Module')
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--min_learning_rate', default=1e-7, type=float)
    parser.add_argument('--lr_decay_steps', default=0, type=int)
    # lr decay的时候会依赖total_steps，这里设置的是total_steps的比例，比如我只需要前50%步做decay，ratio设置为0.5
    parser.add_argument('--lr_decay_ratio', default=1.0, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument('--adam_beta1', default=0.9, type=float)
    parser.add_argument('--adam_beta2', default=0.999, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--scheduler_type', default='polynomial', type=str)
    return parent_args


def get_default_update_params(pl_model: LightningModule):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.', 'layernorm.']
    optimizer_grouped_params = [
        {'params': [p for n, p in pl_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': pl_model.hparams.weight_decay},
        {'params': [p for n, p in pl_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_params


def configure_optimizers(pl_model: LightningModule, model_params=None):
    '''
    Args:
        pl_model： lightning module
        model_params: 需要优化的模型参数
    '''
    # get params that optimizer need
    if model_params is None:
        optimizer_grouped_params = get_default_update_params(pl_model)
    else:
        optimizer_grouped_params = model_params
    # Configure optimizer.
    if isinstance(pl_model.trainer.strategy, DeepSpeedStrategy):
        if 'offload_optimizer' in pl_model.trainer.strategy.config['zero_optimization']:
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
    total_steps = pl_model.hparams.lr_decay_ratio * \
        pl_model.total_steps if pl_model.hparams.lr_decay_steps == 0 else pl_model.hparams.lr_decay_steps
    scheduler = get_scheduler(name=pl_model.hparams.scheduler_type, optimizer=optimizer,
                              num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                              lr_end=pl_model.hparams.min_learning_rate)
    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return [optimizer], [scheduler]


def get_total_steps(trainer, hparams):
    train_loader = trainer._data_connector._train_dataloader_source.dataloader()
    # Calculate total steps
    if trainer.max_epochs > 0:
        world_size = trainer.world_size
        tb_size = hparams.train_batchsize * max(1, world_size)
        ab_size = trainer.accumulate_grad_batches
        total_steps = (len(train_loader.dataset) *
                       trainer.max_epochs // tb_size) // ab_size
    else:
        total_steps = trainer.max_steps
    return total_steps


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    lr_end: Optional[float] = None
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
                             num_training_steps=num_training_steps, lr_end=lr_end)

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

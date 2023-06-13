from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers.optimization import AdamW, TYPE_TO_SCHEDULER_FUNCTION
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers.trainer_utils import SchedulerType
from typing import Optional, Union
import warnings
import types


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


def add_inverse_square_args(parent_args):
    parser = parent_args.add_argument_group('Basic Module')
    parser.add_argument('--warmup_min_lr', default=1e-9, type=float)
    parser.add_argument('--warmup_max_lr', default=1e-4, type=float)

    return parent_args


def get_default_update_params(pl_model: LightningModule):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.', 'layernorm.']
    optimizer_grouped_params = [
        {'params': [p for n, p in pl_model.named_parameters() if not any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': pl_model.hparams.weight_decay},
        {'params': [p for n, p in pl_model.named_parameters() if any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
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
    # elif isinstance(pl_model.trainer.strategy, ColossalAIStrategy):
    #     from colossalai.nn.optimizer import HybridAdam
    #     optimizer = HybridAdam(
    #         optimizer_grouped_params,
    #         lr=pl_model.hparams.learning_rate,
    #         betas=(pl_model.hparams.adam_beta1, pl_model.hparams.adam_beta2),
    #         eps=pl_model.hparams.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_params, lr=pl_model.hparams.learning_rate,
                          betas=(pl_model.hparams.adam_beta1, pl_model.hparams.adam_beta2),
                          eps=pl_model.hparams.adam_epsilon)
    # Configure learning rate scheduler.
    total_steps = pl_model.hparams.lr_decay_ratio * \
        pl_model.total_steps if pl_model.hparams.lr_decay_steps == 0 else pl_model.hparams.lr_decay_steps
    warmup_steps = pl_model.hparams.warmup_ratio * \
        pl_model.total_steps if pl_model.hparams.warmup_steps == 0 else pl_model.hparams.warmup_steps

    if pl_model.hparams.scheduler_type == "inverse_sqrt":
        scheduler = inverse_square_root_schedule(optimizer=optimizer,
                                                 num_warmup_steps=warmup_steps, lr_min=pl_model.hparams.warmup_min_lr, lr_max=pl_model.hparams.warmup_max_lr)
    else:
        scheduler = get_scheduler(name=pl_model.hparams.scheduler_type, optimizer=optimizer,
                                  num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                                  lr_end=pl_model.hparams.min_learning_rate)
    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return [optimizer], [scheduler]


def inverse_square_root_schedule(
        optimizer: Optimizer,
        num_warmup_steps: int = 4000,
        lr_min=1e-9,
        lr_max=1e-4,
        power=0.5,
        last_epoch: int = -1):

    lr_init = optimizer.defaults["lr"]
    if (lr_min > lr_max):
        raise ValueError(f"lr_min ({lr_min}) must be be smaller than lr_max ({lr_max})")

    lr_step = (lr_max - lr_init) / num_warmup_steps
    decay_factor = lr_max * num_warmup_steps**power

    def lr_lambda(current_step: int):
        # 自定义函数
        if current_step < num_warmup_steps:
            return lr_step * current_step
        return decay_factor * current_step ** (-power)

    return Direct_LR(optimizer, lr_lambda, last_epoch, True)


class Direct_LR(_LRScheduler):
    """
    Modified from LambdaLR
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1, warmup_steps=4000, verbose=False):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(Direct_LR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self._step_count < self.warmup_steps:
            return [base_lr + lmbda(self.last_epoch)
                    for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


def get_total_steps(trainer, hparams):
    train_loader = trainer._data_connector._train_dataloader_source.dataloader()
    # Calculate total steps
    if trainer.max_epochs > 0:
        if hasattr(hparams, 'use_mpu'):
            from fengshen.models.megatron import mpu
            world_size = mpu.get_data_parallel_world_size() if hparams.use_mpu else trainer.world_size
        else:
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

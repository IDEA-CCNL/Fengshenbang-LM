from pytorch_lightning import LightningModule

from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers.optimization import AdamW, get_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import warnings
import types


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


def add_inverse_square_args(parent_args):
    parser = parent_args.add_argument_group('Basic Module')
    parser.add_argument('--warmup_min_lr', default=1e-9, type=float)
    parser.add_argument('--warmup_max_lr', default=1e-4, type=float)

    return parent_args


def configure_optimizers(pl_model: LightningModule):
    # Configure learning rate scheduler.
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.', 'layernorm.']
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

    if pl_model.hparams.scheduler_type == "inverse_sqrt":
        scheduler = inverse_square_root_schedule(optimizer=optimizer,
                                                 num_warmup_steps=warmup_steps, lr_min=pl_model.hparams.warmup_min_lr, lr_max=pl_model.hparams.warmup_max_lr)
    else:
        scheduler = get_scheduler(name=pl_model.hparams.scheduler_type, optimizer=optimizer,
                                  num_warmup_steps=warmup_steps, num_training_steps=pl_model.total_steps)
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

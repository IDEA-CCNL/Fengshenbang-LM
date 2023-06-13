# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins import ClusterEnvironment
from pytorch_lightning.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from pytorch_lightning.utilities.types import _PATH, LRSchedulerTypeUnion
from pytorch_lightning.utilities.optimizer import optimizers_to_device
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy as OriginDeepSpeedStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from fengshen.models.megatron import mpu, fused_kernels

log = logging.getLogger(__name__)

if _DEEPSPEED_AVAILABLE:
    import deepspeed


def remove_module_hooks(model: torch.nn.Module) -> None:
    # todo (tchaton) awaiting this feature to move upstream to DeepSpeed
    for module in model.modules():
        module._backward_hooks = OrderedDict()
        module._is_full_backward_hook = None
        module._forward_hooks = OrderedDict()
        module._forward_pre_hooks = OrderedDict()
        module._state_dict_hooks = OrderedDict()
        module._load_state_dict_pre_hooks = OrderedDict()


class DeepSpeedStrategy(OriginDeepSpeedStrategy):
    strategy_name = "megatron_deepspeed"
    DEEPSPEED_ENV_VAR = "PL_DEEPSPEED_CONFIG_PATH"

    def __init__(
        self,
        pipe_model_parallel_size,
        tensor_model_parallel_size,
        mpu_seed,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        zero_optimization: bool = True,
        stage: int = 2,
        remote_device: str = "cpu",
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        offload_params_device: str = "cpu",
        nvme_path: str = "/local_nvme",
        params_buffer_count: int = 5,
        params_buffer_size: int = 100_000_000,
        max_in_cpu: int = 1_000_000_000,
        offload_optimizer_device: str = "cpu",
        optimizer_buffer_count: int = 4,
        block_size: int = 1048576,
        queue_depth: int = 8,
        single_submit: bool = False,
        overlap_events: bool = True,
        thread_count: int = 1,
        pin_memory: bool = False,
        sub_group_size: int = 1_000_000_000_000,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 200_000_000,
        reduce_bucket_size: int = 200_000_000,
        zero_allow_untested_optimizer: bool = True,
        logging_batch_size_per_gpu: Union[str, int] = "auto",
        config: Optional[Union[_PATH, Dict[str, Any]]] = None,
        logging_level: int = logging.WARN,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        loss_scale: float = 0,
        initial_scale_power: int = 16,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1,
        partition_activations: bool = False,
        cpu_checkpointing: bool = False,
        contiguous_memory_optimization: bool = False,
        synchronize_checkpoint_boundary: bool = False,
        load_full_weights: bool = False,
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = None,
    ) -> None:
        """Provides capabilities to run training using the DeepSpeed library, with training optimizations for large
        billion parameter models. `For more information: https://pytorch-
        lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed`.

        .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

        Defaults have been set to enable ZeRO-Offload and some have been taken from the link below.
        These defaults have been set generally, but may require tuning for optimum performance based on your model size.
        `For more information: https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training`.

        Arguments:

            zero_optimization: Enable ZeRO optimization. This is compatible with either `precision="16-mixed"` or
                `precision="bf16-mixed"`.

            stage: Different stages of the ZeRO Optimizer. 0 is disabled,
                1 is optimizer state partitioning, 2 is optimizer+gradient state partitioning,
                3 is optimizer+gradient_parameter partitioning using the infinity engine.

            remote_device: Device to instantiate the model on initially (``cpu`` or ``nvme``).

            offload_optimizer: Enable offloading optimizer memory and computation to CPU or NVMe
                based on ``offload_optimizer_device``.

            offload_parameters: When using ZeRO Stage 3, Enable offloading parameter memory and computation
                to CPU or NVMe based on ``offload_params_device``.

            offload_params_device: When offloading parameters choose the device to offload to, ``cpu`` or ``nvme``.

            offload_optimizer_device: When offloading optimizer state choose the device to offload to,
                ``cpu`` or ``nvme``.

            params_buffer_count: Number of buffers in buffer pool for
                parameter offloading when ``offload_params_device`` is ``nvme``.

            params_buffer_size: Size of buffers in buffer pool for parameter offloading
                when ``offload_params_device`` is ``nvme``.

            max_in_cpu: Number of parameter elements to maintain in CPU memory when offloading to NVMe is enabled.

            nvme_path: Filesystem path for NVMe device for optimizer/parameter state offloading.

            optimizer_buffer_count: Number of buffers in buffer pool for optimizer state offloading
                when ``offload_optimizer_device`` is set to to ``nvme``.
                This should be at least the number of states maintained per parameter by the optimizer.
                For example, Adam optimizer has 4 states (parameter, gradient, momentum, and variance).

            block_size: When using NVMe Offloading, the I/O block size in bytes.

            queue_depth: When using NVMe Offloading, the I/O queue depth.

            single_submit: When using NVMe Offloading,
                submit requests to storage device as multiple individual requests,
                as opposed to one block of requests.

            overlap_events: When using NVMe Offloading,
                submit requests to storage device in an overlapped fashion
                without waiting for completion of earlier requests.

            thread_count: When using NVMe Offloading,
                Intra-request parallelism for each read/write submitted by a user thread.

            pin_memory: When using ZeRO stage 3, pin optimizer state memory on CPU.
                This could boost throughput at the cost of extra memory overhead.

            sub_group_size: When using ZeRO stage 3, defines the number of parameters
                within a sub group to offload at a time.
                Smaller numbers require more communication, but improve memory efficiency.

            contiguous_gradients: Copies gradients to a continuous buffer as they are produced.
                Avoids memory fragmentation during backwards. Useful when training large models.

            overlap_comm: Overlap the reduction (synchronization) of gradients with the backwards computation.
                This is a speed optimization when training across multiple GPUs/machines.

            allgather_partitions: All gather updated parameters at the end of training step,
                instead of using a series of broadcast collectives.

            reduce_scatter: Use reduce/scatter instead of allreduce to average gradients.

            allgather_bucket_size: Number of elements to allgather at once.
                Used to limit the memory required for larger model sizes, with a tradeoff with speed.

            reduce_bucket_size: Number of elements to reduce at once.
                Used to limit the memory required for larger model sizes, with a tradeoff with speed.

            zero_allow_untested_optimizer: Allow untested optimizers to be used with ZeRO. Currently only Adam is a
                DeepSpeed supported optimizer when using ZeRO.

            logging_batch_size_per_gpu: Config used in DeepSpeed to calculate verbose timing for logging
                on a per sample per second basis (only displayed if logging=logging.INFO).
                If set to "auto", the plugin tries to infer this from
                the train DataLoader's BatchSampler, else defaults to 1.
                To obtain accurate logs when using datasets that do not support batch samplers,
                set this to the actual per gpu batch size (trainer.batch_size).

            config: Pass in a deepspeed formatted config dict,
                or path to a deepspeed config: https://www.deepspeed.ai/docs/config-json.
                All defaults will be ignored if a config is passed in.

            logging_level: Set logging level for deepspeed.

            loss_scale: Loss scaling value for FP16 training.
                0.0 results in dynamic loss scaling, otherwise static.

            initial_scale_power: Power of the initial dynamic loss scale value. Loss scale is computed
                by ``2^initial_scale_power``.

            loss_scale_window: Window in which to raise/lower the dynamic FP16 loss scaling value.

            hysteresis: FP16 Delay shift in Dynamic Loss scaling.

            min_loss_scale: The minimum FP16 dynamic loss scaling value.

            partition_activations: Enables partition activation when used with ZeRO stage 3 and model parallelism.
                Still requires you to wrap your forward functions in deepspeed.checkpointing.checkpoint.
                See `deepspeed tutorial
                <https://www.deepspeed.ai/tutorials/megatron/#deepspeed-activation-checkpoints-optional>`_.

            cpu_checkpointing: Offloads partitioned activations to CPU if ``partition_activations`` is enabled.

            contiguous_memory_optimization: Copies partitioned activations so that they are contiguous in memory.
                Not supported by all models.

            synchronize_checkpoint_boundary: Insert :func:`torch.cuda.synchronize` at each checkpoint boundary.

            load_full_weights: True when loading a single checkpoint file containing the model state dict
                when using ZeRO Stage 3. This differs from the DeepSpeed checkpoint which contains shards
                per worker.
        """
        if not _DEEPSPEED_AVAILABLE:
            raise MisconfigurationException(
                "To use the `DeepSpeedStrategy`, you must have DeepSpeed installed."
                " Install it by running `pip install -U deepspeed`."
            )

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
        )

        self.config = self._load_config(config)
        if self.config is None:
            # User has not overridden config, set defaults
            self.config = self._create_default_config(
                zero_optimization,
                zero_allow_untested_optimizer,
                logging_batch_size_per_gpu,
                offload_optimizer=offload_optimizer,
                offload_parameters=offload_parameters,
                nvme_path=nvme_path,
                offload_params_device=offload_params_device,
                params_buffer_count=params_buffer_count,
                params_buffer_size=params_buffer_size,
                max_in_cpu=max_in_cpu,
                pin_memory=pin_memory,
                offload_optimizer_device=offload_optimizer_device,
                optimizer_buffer_count=optimizer_buffer_count,
                block_size=block_size,
                queue_depth=queue_depth,
                single_submit=single_submit,
                overlap_events=overlap_events,
                thread_count=thread_count,
                partition_activations=partition_activations,
                cpu_checkpointing=cpu_checkpointing,
                contiguous_memory_optimization=contiguous_memory_optimization,
                synchronize_checkpoint_boundary=synchronize_checkpoint_boundary,
                stage=stage,
                contiguous_gradients=contiguous_gradients,
                overlap_comm=overlap_comm,
                allgather_partitions=allgather_partitions,
                reduce_scatter=reduce_scatter,
                allgather_bucket_size=allgather_bucket_size,
                reduce_bucket_size=reduce_bucket_size,
                sub_group_size=sub_group_size,
            )
        import deepspeed

        self._config_initialized = False
        deepspeed.utils.logging.logger.setLevel(logging_level)

        self.remote_device = remote_device
        self.load_full_weights = load_full_weights

        # default FP16 parameters.
        self.loss_scale = loss_scale
        self.initial_scale_power = initial_scale_power
        self.loss_scale_window = loss_scale_window
        self.hysteresis = hysteresis
        self.min_loss_scale = min_loss_scale
        self.pipe_model_parallel_size = pipe_model_parallel_size
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.mpu_seed = mpu_seed

    def _setup_model_and_optimizer(
        self, model: Module, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
    ):
        """Initialize one model and one optimizer with an optional learning rate scheduler.

        This calls :func:`deepspeed.initialize` internally.
        """
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=self.root_device.index),
            config=self.config,
            model=model,
            model_parameters=model_parameters,  # type: ignore
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=False,
            mpu=mpu
        )
        return deepspeed_engine, deepspeed_optimizer

    def _set_deepspeed_activation_checkpointing(self) -> None:
        import deepspeed

        assert isinstance(self.config, dict)
        assert self.config.get(
            "activation_checkpointing"), 'megatron_deepspeed stratygy need activation_checkpointing config'
        if self.config.get("activation_checkpointing"):
            checkpoint_config = self.config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=mpu,
                num_checkpoints=checkpoint_config.get("num_checkpoints"),
                partition_activations=checkpoint_config.get("partition_activations"),
                contiguous_checkpointing=checkpoint_config.get("contiguous_memory_optimization"),
                checkpoint_in_cpu=checkpoint_config.get("cpu_checkpointing"),
                profile=checkpoint_config.get("profile"),
            )

    def setup_environment(self) -> None:
        super().setup_environment()
        self.setup_mpu()

    def setup_mpu(self) -> None:
        fused_kernels.load_fused_kernels()
        rank = self.cluster_environment.global_rank()
        world_size = self.cluster_environment.world_size()
        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

        # this does pipe on the most outside, then data, then model.
        # PipeModelDataParallelTopology is just a wrapper over ProcessTopology that predefines this order.
        dp = world_size // self.pipe_model_parallel_size // self.tensor_model_parallel_size
        topo = PipeModelDataParallelTopology(num_pp=self.pipe_model_parallel_size,
                                             num_mp=self.tensor_model_parallel_size,
                                             num_dp=dp)

        # Offset base seeds for the interior pipeline stages.
        # TODO: adjust last stage too once IO is improved.
        stage_id = topo.get_coord(rank=rank).pipe
        if 0 < stage_id < topo.get_dim("pipe") - 1:
            offset = seed + 1138
            seed = offset + (stage_id * self.tensor_model_parallel_size)

        mpu.initialize_model_parallel(
            self.tensor_model_parallel_size,
            topology=topo,
            fp32_allreduce=False)

        self._set_deepspeed_activation_checkpointing()
        mpu.model_parallel_cuda_manual_seed(self.mpu_seed)
        
    def _initialize_deepspeed_inference(self, model: Module) -> None:
        import deepspeed

        assert isinstance(self.config, dict)

        # todo: this is required for DeepSpeed throughput timers
        inference_config = {"train_micro_batch_size_per_gpu": 1}
        if "fp16" in self.config:
            inference_config.update({"fp16": self.config["fp16"]})
        if self.zero_stage_3:
            inference_config.update(
                {
                    "zero_allow_untested_optimizer": self.config["zero_allow_untested_optimizer"],
                    "zero_optimization": self.config["zero_optimization"],
                }
            )
        # Remove all module hooks before initializing new model
        remove_module_hooks(model)
        model, _, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=self.root_device.index),
            config=inference_config,
            model=model,
            optimizer=None,
            lr_scheduler=None,
            model_parameters=[],
            dist_init_required=False,
            mpu=mpu
        )
        self.model = model

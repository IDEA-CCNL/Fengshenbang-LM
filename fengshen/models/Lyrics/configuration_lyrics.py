# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" BLIP-2 model configuration"""

import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

# BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "salesforce/blip2-opt-2.7b": "https://huggingface.co/salesforce/blip2-opt-2.7b/resolve/main/config.json",
# }


class LyricsVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Blip2VisionModel`]. It is used to instantiate a
    BLIP-2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the BLIP-2
    [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 39):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported. layer_norm_eps (`float`, *optional*, defaults
            to 1e-5): The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries and values in the self-attention layers.

    Example:

    ```python
    >>> from transformers import Blip2VisionConfig, Blip2VisionModel

    >>> # Initializing a Blip2VisionConfig with Salesforce/blip2-opt-2.7b style configuration
    >>> configuration = Blip2VisionConfig()

    >>> # Initializing a Blip2VisionModel (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
    >>> model = Blip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_2_vision_model"

    def __init__(
        self,
        hidden_size=1408,
        intermediate_size=6144,
        num_hidden_layers=39,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=0.00001,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
    
class LyricsDetectionConfig(PretrainedConfig):

    model_type = "blip_2_detection_model"

    def __init__(
        self,
        backbone = "swin_T_224_1k",
        position_embedding = "sine",
        pe_temperatureh = 20,
        pe_temperaturew = 20,
        return_interm_indices = [1, 2, 3],
        backbone_freeze_keywords = None,
        enc_layers = 6,
        num_unicoder_layers = 0,
        dec_layers = 6,
        pre_norm = False,
        dim_feedforward = 2048,
        hidden_dim = 256,
        dropout = 0.0,
        nheads = 8,
        num_queries = 900,
        aux_loss = True,
        iter_update = True,
        dn_number = 0,
        query_dim = 4,
        num_patterns = 0,
        num_feature_levels = 4,
        enc_n_points = 4,
        dec_n_points = 4,
        learnable_tgt_init = True,
        two_stage_type = "standard",
        two_stage_bbox_embed_share = False,
        two_stage_class_embed_share = False,
        transformer_activation = "relu",
        return_intermediate_dec = True,
        dec_pred_bbox_embed_share = True,
        dn_box_noise_scale = 1.0,
        dn_label_noise_ratio = 0.5,
        dn_label_coef = 1.0,
        dn_bbox_coef = 1.0,
        embed_init_tgt = True,
        dn_labelbook_size = 2000,
        max_text_len = 256,
        text_encoder_type = "bert-base-uncased",
        use_checkpoint = True,
        use_transformer_ckpt = True,
        use_text_cross_attention = True,
        text_dropout = 0.0,
        fusion_dropout = 0.0,
        fusion_droppath = 0.1,
        sub_sentence_present = True,
        pretrain_img_size = 224,
        patch_size = 4,
        in_chans = 3,
        num_layers = 4,
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24],
        window_size = 7,
        mlp_ratio = 4.0,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0.0,
        attn_drop_rate = 0.0,
        drop_path_rate = 0.2,
        swintransformer_use_checkpoint = False,
        ape = False,
        patch_norm = True,
        out_indices = [1, 2, 3],
        frozen_stages = -1,
        dilation = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.position_embedding = position_embedding
        self.pe_temperatureh = pe_temperatureh
        self.pe_temperaturew = pe_temperaturew
        self.return_interm_indices = return_interm_indices
        self.backbone_freeze_keywords = backbone_freeze_keywords
        self.enc_layers = enc_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.dec_layers = dec_layers
        self.pre_norm = pre_norm
        self.dim_feedforward = dim_feedforward
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.nheads = nheads
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.iter_update = iter_update
        self.dn_number = dn_number
        self.query_dim = query_dim
        self.num_patterns = num_patterns
        self.num_feature_levels = num_feature_levels
        self.enc_n_points = enc_n_points
        self.dec_n_points = dec_n_points
        self.learnable_tgt_init = learnable_tgt_init
        self.two_stage_type = two_stage_type
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.two_stage_class_embed_share = two_stage_class_embed_share
        self.transformer_activation = transformer_activation
        self.return_intermediate_dec = return_intermediate_dec
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_label_coef = dn_label_coef
        self.dn_bbox_coef = dn_bbox_coef
        self.embed_init_tgt = embed_init_tgt
        self.dn_labelbook_size = dn_labelbook_size
        self.max_text_len = max_text_len
        self.text_encoder_type = text_encoder_type
        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt
        self.use_text_cross_attention = use_text_cross_attention
        self.text_dropout = text_dropout
        self.fusion_dropout = fusion_dropout
        self.fusion_droppath = fusion_droppath
        self.sub_sentence_present = sub_sentence_present
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.swintransformer_use_checkpoint = swintransformer_use_checkpoint
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dilation = dilation

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["detection_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        
        return cls.from_dict(config_dict, **kwargs)
    
class LyricsRAMConfig(PretrainedConfig):

    model_type = "blip_2_ram_model"

    def __init__(
        self,
        med_config='/med_config.json',
        image_size=384,
        window_size=12,
        vit='swin_l',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        prompt='a picture of ',
        threshold=0.68,
        delete_tag_index=[],
        tag_list='/ram_tag_list.txt',
        tag_list_chinese='/ram_tag_list_chinese.txt',
        vision_width=1536,
        image_res=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.med_config = med_config
        self.image_size = image_size
        self.window_size = window_size
        self.vit = vit
        self.vit_grad_ckpt = vit_grad_ckpt
        self.vit_ckpt_layer = vit_ckpt_layer
        self.prompt = prompt
        self.threshold = threshold
        self.delete_tag_index = delete_tag_index
        self.tag_list = tag_list
        self.tag_list_chinese = tag_list_chinese
        self.vision_width = vision_width
        self.image_res = image_res
        self.embed_dim =embed_dim
        self.depths = depths
        self.num_heads = num_heads

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["ram_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)    


class LyricsQFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Blip2QFormerModel`]. It is used to instantiate a
    BLIP-2 Querying Transformer (Q-Former) model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the BLIP-2
    [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture. Configuration objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Note that [`Blip2QFormerModel`] is very similar to [`BertLMHeadModel`] with interleaved cross-attention.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Q-Former model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling the model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        cross_attention_frequency (`int`, *optional*, defaults to 2):
            The frequency of adding cross-attention to the Transformer layers.
        encoder_hidden_size (`int`, *optional*, defaults to 1408):
            The hidden size of the hidden states for cross-attention.

    Examples:

    ```python
    >>> from transformers import Blip2QFormerConfig, Blip2QFormerModel

    >>> # Initializing a BLIP-2 Salesforce/blip2-opt-2.7b style configuration
    >>> configuration = Blip2QFormerConfig()

    >>> # Initializing a model (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
    >>> model = Blip2QFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "blip_2_qformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        cross_attention_frequency=2,
        encoder_hidden_size=1408,
        detection_encoder_hidden_size=256,
        query_length=96,
        num_vit_query_tokens=32,
        num_dino_query_tokens=64,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.cross_attention_frequency = cross_attention_frequency
        self.encoder_hidden_size = encoder_hidden_size
        self.detection_encoder_hidden_size = detection_encoder_hidden_size
        self.query_length = query_length
        self.num_vit_query_tokens = num_vit_query_tokens
        self.num_dino_query_tokens = num_dino_query_tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["qformer_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class LyricsConfig(PretrainedConfig):
    r"""
    [`Blip2Config`] is the configuration class to store the configuration of a [`Blip2ForConditionalGeneration`]. It is
    used to instantiate a BLIP-2 model according to the specified arguments, defining the vision model, Q-Former model
    and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Blip2VisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Blip2QFormerConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     Blip2VisionConfig,
    ...     Blip2QFormerConfig,
    ...     OPTConfig,
    ...     Blip2Config,
    ...     Blip2ForConditionalGeneration,
    ... )

    >>> # Initializing a Blip2Config with Salesforce/blip2-opt-2.7b style configuration
    >>> configuration = Blip2Config()

    >>> # Initializing a Blip2ForConditionalGeneration (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
    >>> model = Blip2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Blip2Config from a Blip2VisionConfig, Blip2QFormerConfig and any PretrainedConfig

    >>> # Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
    >>> vision_config = Blip2VisionConfig()
    >>> qformer_config = Blip2QFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = Blip2Config.from_text_vision_configs(vision_config, qformer_config, text_config)
    ```"""

    model_type = "blip-2"
    is_composition = True

    def __init__(self, vision_config=None, qformer_config=None, text_config=None, detection_config=None, ram_config=None, num_query_tokens=96, image_text_hidden_size=256,**kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the LyricsVisionConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the LyricsQFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")

        if detection_config is None:
            detection_config = {}
            logger.info("detection_config is None. initializing the LyricsDetectionConfig with default values.")

        if ram_config is None:
            ram_config = {}
            logger.info("ram_config is None. Initializing the LyricsRAMConfig with default values.")

        self.vision_config = LyricsVisionConfig(**vision_config)
        self.qformer_config = LyricsQFormerConfig(**qformer_config)
        self.detection_config = LyricsDetectionConfig(**detection_config)
        self.ram_config = LyricsRAMConfig(**ram_config)
        text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.num_query_tokens = num_query_tokens
        self.image_text_hidden_size = image_text_hidden_size
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: LyricsVisionConfig,
        qformer_config: LyricsQFormerConfig,
        text_config: PretrainedConfig,
        detection_config: LyricsDetectionConfig,
        ram_config: LyricsRAMConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Blip2Config`] (or a derived class) from a BLIP-2 vision model, Q-Former and language model
        configurations.

        Returns:
            [`Blip2Config`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            text_config=text_config.to_dict(),
            detection_config=detection_config.to_dict(),
            ram_config=ram_config.to_dict(),
            **kwargs,
        )
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["qformer_config"] = self.qformer_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["detection_config"] = self.detection_config.to_dict()
        output["ram_config"] = self.ram_config.to_dict()        
        output["model_type"] = self.__class__.model_type
        return output

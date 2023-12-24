# coding=utf-8
# Copyright 2023 The Salesforce Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch BLIP-2 model."""

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List
import warnings
import random
import torchvision
import torch
import torch.utils.checkpoint
import copy
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.distributed as dist
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
)
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    logging,
)
# from transformers.models.blip_2.configuration_blip_2 import Blip2Config, Blip2QFormerConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput
from transformers import (
    Blip2PreTrainedModel,
    Blip2VisionModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Blip2QFormerModel,
    PreTrainedTokenizer,
    LogitsProcessorList,
    LogitsProcessor,
    StoppingCriteriaList,
    GenerationConfig,
)
from fengshen.models.Lyrics.groundingdino.modeling_groundingdino import GroundingDINO
from fengshen.models.Lyrics.ram.models.ram import RAM
from fengshen.models.Lyrics.configuration_lyrics import LyricsConfig, LyricsQFormerConfig


logger = logging.get_logger(__name__)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # transformer为layernorm, lavis为LayerNorm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.config = config

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length: seq_length + past_key_values_length
            ].clone()
        # print(position_ids)

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LyricsQFormerMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, is_detection=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention and is_detection:
            self.key = nn.Linear(config.detection_encoder_hidden_size, self.all_head_size) # 260, 256 + 4
            self.value = nn.Linear(config.detection_encoder_hidden_size, self.all_head_size)
        elif is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)            
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            if self.key.weight.dtype == torch.half:
                encoder_hidden_states = encoder_hidden_states.half()
            # encoder_hidden_states = encoder_hidden_states
            elif self.key.weight.dtype == torch.bfloat16:
                encoder_hidden_states = torch.tensor(encoder_hidden_states, dtype=torch.bfloat16)
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long,
                                          device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long,
                                          device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Blip2QFormer
class LyricsQFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LyricsQFormerAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, is_detection = False):
        super().__init__()
        self.attention = LyricsQFormerMultiHeadAttention(config, is_cross_attention, is_detection)
        self.output = LyricsQFormerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Blip2QFormer
class LyricsQFormerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Blip2QFormer
class LyricsQFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LyricsQFormerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LyricsQFormerAttention(config)

        self.layer_idx = layer_idx
        self.num_vit_query_tokens = config.num_vit_query_tokens

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = LyricsQFormerAttention(config, is_cross_attention=True)
            self.detection_crossattention = LyricsQFormerAttention(config, is_cross_attention=True, is_detection=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = LyricsQFormerIntermediate(config)
        self.output = LyricsQFormerOutput(config)

        self.intermediate_query = LyricsQFormerIntermediate(config)
        self.output_query = LyricsQFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        detection_encoder_hidden_states=None,
        detection_encoder_attention_mask=None,        
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError(
                        "encoder_hidden_states must be given for cross-attention layers")
                if detection_encoder_hidden_states is None:
                    raise ValueError(
                        "detection_encoder_hidden_states must be given for cross-attention layers")                
                if attention_mask is not None:
                    cross_attention_mask = attention_mask[:, :self.num_vit_query_tokens]
                    detection_cross_attention_mask = attention_mask[:, self.num_vit_query_tokens:]
                else:
                    cross_attention_mask = None
                    detection_cross_attention_mask = None
                cross_attention_outputs = self.crossattention(
                    query_attention_output[:, :self.num_vit_query_tokens, :],
                    cross_attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                vit_query_attention_output = cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                query_attention_probs = cross_attention_outputs[1:-1]

                detection_cross_attention_outputs = self.detection_crossattention(
                    query_attention_output[:, self.num_vit_query_tokens:, :],
                    detection_cross_attention_mask,
                    head_mask,
                    detection_encoder_hidden_states,
                    detection_encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                detection_query_attention_output = detection_cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                detection_query_attention_probs = detection_cross_attention_outputs[1:-1]
                
                if output_attentions == True:
                    padding_attention = torch.zeros((query_attention_probs[0].size(0),
                                                    query_attention_probs[0].size(1),
                                                    detection_query_attention_probs[0].size(2) - query_attention_probs[0].size(2)))
                    query_attention_probs = torch.cat([query_attention_probs[0], padding_attention], dim = -1)

                    outputs = outputs + (torch.cat([query_attention_probs[0], detection_query_attention_probs], dim=1),)
                else:
                    outputs = outputs + cross_attention_outputs[1:-1]

                query_attention_output = torch.cat([vit_query_attention_output, detection_query_attention_output], dim=1)

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        # present_key_value是self attention的key,value, 用于在decoder中以前的词的key,value不用再重复计算
        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class LyricsQFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [LyricsQFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        detection_encoder_hidden_states=None,
        detection_encoder_attention_mask=None,        
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):

            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    detection_encoder_hidden_states,
                    detection_encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    detection_encoder_hidden_states,
                    detection_encoder_attention_mask,                    
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 这里的cross attention是经过pad的
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class LyricsQFormerModel(Blip2PreTrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: LyricsQFormerConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        # self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = LyricsQFormerEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        is_decoder: bool,
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )

                # add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    if has_query:  # UniLM style attention mask
                        causal_mask = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prefix_seq_len, seq_length),
                                    device=device,
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask,
                            ],
                            axis=1,
                        )
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, causal_mask.shape[1], prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        query_embeds=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        detection_encoder_hidden_states=None,
        detection_encoder_attention_mask=None,        
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            assert (
                query_embeds is not None
            ), "You have to specify query_embeds when input_ids is None"

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] -
            self.config.query_length if past_key_values is not None else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeds=query_embeds,
            past_key_values_length=past_key_values_length,
        )

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device
        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                input_ids.shape,
                device,
                is_decoder,
                has_query=(query_embeds is not None),
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, device, is_decoder
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if detection_encoder_hidden_states is not None:
            if type(detection_encoder_hidden_states) == list:
                detection_encoder_batch_size, detection_encoder_sequence_length, _ = detection_encoder_hidden_states[0].size()
            else:
                (
                    detection_encoder_batch_size,
                    detection_encoder_sequence_length,
                    _,
                ) = detection_encoder_hidden_states.size()
            detection_encoder_hidden_shape = (detection_encoder_batch_size, detection_encoder_sequence_length)

            if type(detection_encoder_attention_mask) == list:
                detection_encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in detection_encoder_attention_mask]
            elif detection_encoder_attention_mask is None:
                detection_encoder_attention_mask = torch.ones(detection_encoder_hidden_shape, device=device)
                detection_encoder_extended_attention_mask = self.invert_attention_mask(detection_encoder_attention_mask)
            else:
                detection_encoder_extended_attention_mask = self.invert_attention_mask(detection_encoder_attention_mask)
        else:
            detection_encoder_extended_attention_mask = None
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            detection_encoder_hidden_states=detection_encoder_hidden_states,
            detection_encoder_attention_mask=detection_encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->BlipText


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config: LyricsQFormerConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

# 把注意力方式改一下，就可以做MLM了
class LyricsQFormerOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LyricsQFormerWithLMHead(Blip2PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config: LyricsQFormerConfig):
        super().__init__(config)

        self.bert = LyricsQFormerModel(config)
        self.cls = LyricsQFormerOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        detection_encoder_hidden_states=None,
        detection_encoder_attention_mask=None,        
        labels=None,
        past_key_values=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=True,
        reduction="mean",
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            detection_encoder_hidden_states=detection_encoder_hidden_states,
            detection_encoder_attention_mask=detection_encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1]:, :]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        # 区分is_decoder
        # 没mask掉的用-100代替
        lm_loss = None
        if is_decoder == True:
            if labels is not None:
                # we are doing next-token prediction; shift prediction scores and input ids by one
                shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()
                loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
                lm_loss = loss_fct(
                    shifted_prediction_scores.view(-1, self.config.vocab_size),
                    labels.view(-1),
                )
                if reduction == "none":
                    lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)
            
            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((lm_loss,) + output) if lm_loss is not None else output
        else:
            if labels is not None:
                # we are doing mask prediction; do not need shift; but do not calculate cls_token
                # bs, seq, vocab
                prediction_scores = prediction_scores[:, 1:, :].contiguous()
                # bs, seq
                labels = labels[:, 1:].contiguous()
                # print('prediction_scores:', prediction_scores.size())
                # print('labels:', labels.size())
                # print('max_labels:', torch.max(labels, dim=1))
                loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
                mlm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size),
                    labels.view(-1),
                )
                if reduction == "none":
                    mlm_loss = mlm_loss.view(prediction_scores.size(0), -1).sum(1)            
            
            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((mlm_loss,) + output) if mlm_loss is not None else output

        if is_decoder == True:
            return CausalLMOutputWithCrossAttentions(
                loss=lm_loss,
                logits=prediction_scores,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
        
        else:
            return MaskedLMOutput(
                loss=mlm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past


@dataclass
class LyricsOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None

    loss_itc: Optional[torch.FloatTensor] = None

    loss_itm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None

    loss_mlm: Optional[torch.FloatTensor] = None

# 用来封装BLIPQFormerWithLMHead的输出的
class LyricsQFormerForConditionalGeneration(Blip2PreTrainedModel):
    config_class = LyricsConfig
    main_input_name = "pixel_values"

    def __init__(self, config: LyricsConfig):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.qformer_config.hidden_size))

        self.qformer = LyricsQFormerWithLMHead(config.qformer_config)

        self.decoder_input_ids = config.qformer_config.bos_token_id
        self.decoder_pad_token_id = config.qformer_config.pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(pixel_values.device)

        query_outputs = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        lm_output = self.qformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=return_dict,
            labels=labels,
        )

        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return Blip2ForConditionalGenerationModelOutput(
            loss=lm_output.loss,
            decoder_logits=lm_output.logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=lm_output,
        )

class LyricsLMForConditionalGeneration(Blip2PreTrainedModel):
    config_class = LyricsConfig
    main_input_name = "pixel_values"

    def __init__(self, config: LyricsConfig):
        super().__init__(config)

        # 直接传ids进来，所以不需要tokenizer
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.ram = RAM(config.ram_config)
        self.grounding_dino = GroundingDINO(config.detection_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.qformer_config.hidden_size))
        
        # 只用了encoder那部分模型，没用有cls的模型
        self.qformer = LyricsQFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for",
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        ram_pixel_values: torch.FloatTensor,
        grounding_pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        labels: torch.FloatTensor = None,
        # 因为label不会出现在image之前，所以这里不需要labels_before_image， 按照input_ids_before_image补-100就可以了
        qformer_input_ids: torch.FloatTensor = None,
        qformer_attention_mask: torch.FloatTensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        tags_english, tags_chinese = self.ram.generate_tag(ram_pixel_values)

        input_tags = [tag.replace(' |', ',').lower().strip() + "." if not tag.endswith(".") else tag.replace(' |', ',').lower().strip() for tag in tags_english]

        # outputs = self.grounding_dino(grounding_image[None], captions=input_tags)
        grounding_outputs = self.grounding_dino(grounding_pixel_values, captions=input_tags)

        detection_image_embeds = grounding_outputs["hidden_state"] # (bs, nq, 256)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        detection_image_attention_mask = torch.ones(detection_image_embeds.size()[:-1], dtype=torch.long).to(pixel_values.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)

        if qformer_input_ids == None:
            # print('no_hava_instruct')
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                detection_encoder_hidden_states=detection_image_embeds,
                detection_encoder_attention_mask=detection_image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            query_output = query_outputs[0]            
        else:
            # print('hava_instruct')
            text_qformer_atts = qformer_attention_mask
            qformer_atts = torch.cat([query_atts, text_qformer_atts],dim=1)
            query_outputs = self.qformer(
                qformer_input_ids,
                query_embeds=query_tokens,
                attention_mask=qformer_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                detection_encoder_hidden_states=detection_image_embeds,
                detection_encoder_attention_mask=detection_image_attention_mask,                
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            query_output = query_outputs[0][:,:query_tokens.size(1),:]

        # print(query_output.size())
        # step 2.5 generate the lm input by prompt and output
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        # 确保language_model_inputs的batch
        assert language_model_inputs.shape[0] == input_ids.shape[0]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat(
            [
                language_model_inputs,
                inputs_embeds.to(language_model_inputs.device)
            ], dim=1)

        attention_mask = torch.cat(
            [
                language_model_attention_mask,
                attention_mask.to(language_model_attention_mask.device)
            ], dim=1
        )

        # labels也需要对应的处理，把前面空缺的-100加进去
        if labels is not None:
            labels = torch.cat(
                [
                    torch.tensor([-100]).expand(query_tokens.shape[:-1]
                                                ).to(language_model_inputs.device),
                    labels,
                ], dim=1
            )

        # step 3: use the language model

        if self.config.use_decoder_only_language_model:
            # print('model is a use_decoder_only_language_model')
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # labels=labels,
            )

            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))           
                   

        else:
            raise Exception("not impl")
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        ram_pixel_values: torch.FloatTensor,
        grounding_pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor = None,
        qformer_attention_mask: torch.FloatTensor = None, 
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        # print('data type: ', pixel_values.dtype)
        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

        tags_english, tags_chinese = self.ram.generate_tag(ram_pixel_values)

        input_tags = [tag.replace(' |', ',').lower().strip() + "." if not tag.endswith(".") else tag.replace(' |', ',').lower().strip() for tag in tags_english]

        # outputs = self.grounding_dino(grounding_image[None], captions=input_tags)
        grounding_outputs = self.grounding_dino(grounding_pixel_values, captions=input_tags)

        detection_image_embeds = grounding_outputs["hidden_state"] # (bs, nq, 256)

        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        detection_image_attention_mask = torch.ones(detection_image_embeds.size()[:-1], dtype=torch.long).to(pixel_values.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)

        if qformer_input_ids == None:
            # print('no_hava_instruct')
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                detection_encoder_hidden_states=detection_image_embeds,
                detection_encoder_attention_mask=detection_image_attention_mask,
            )
            query_output = query_outputs[0]          
        else:
            # print('hava_instruct')
            if qformer_attention_mask == None:
                qformer_attention_mask = torch.ones(qformer_input_ids.size(), dtype=torch.long).to(image_embeds.device)
            qformer_atts = torch.cat([query_atts, qformer_attention_mask],dim=1)
            query_outputs = self.qformer(
                qformer_input_ids,
                query_embeds=query_tokens,
                attention_mask=qformer_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                detection_encoder_hidden_states=detection_image_embeds,
                detection_encoder_attention_mask=detection_image_attention_mask,                
            )
            query_output = query_outputs[0][:,:query_tokens.size(1),:]
            # print('query_output:', query_output)
            # print('query_output_size:', query_output.size())

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        # print('language_model_inputs:', language_model_inputs)
        # print('language_model_inputs_size:', language_model_inputs.size())


        if attention_mask == None:
            assert batch_size == 1 , print('If you do not pass in llm_instruct_atts, you can only be generated in a single sentence.')
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask], dim=1)

        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
        # print('inputs_embeds:', inputs_embeds)
        # print('attention_mask:', attention_mask)
        language_outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # outputs = []
        # for index, output in enumerate(language_outputs):
        #     output = output[inputs_embeds[index].size(0):]
        #     outputs.append(output)

        # return outputs
        return language_outputs


class LyricsQFromerForPretrain(Blip2PreTrainedModel):
    config_class = LyricsConfig

    def __init__(self, config: LyricsConfig):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)
        self.ram = RAM(config.ram_config)
        self.grounding_dino = GroundingDINO(config.detection_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.qformer_config.hidden_size))
        
        # 同一个LMhead，不同的任务，加一个参数。或者拼起来。估计要加一个linear
        # 图片256，目标检测900，语义分割4096，怎么可以赋予不同的权重，时间不是问题，权重是问题
        self.qformer = LyricsQFormerWithLMHead(config.qformer_config)

        self.vision_proj = nn.Linear(self.qformer.config.hidden_size, config.image_text_hidden_size)
        self.text_proj = nn.Linear(self.qformer.config.hidden_size, config.image_text_hidden_size)

        self.itm_head = nn.Linear(self.qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = 512 # 512-96 = 416
        self.max_input_len = 600

        # Initialize weights and apply final processing
        self.post_init()

    def generate_bbox_caption(self, logits, boxes, english_tags, chinese_tags, language):
        # filter output
        # 0最大值, 1索引
        bbox_caption = []
        bbox_caption_tokens = []
        bbox_caption_tokens_with_mask = []
        bbox_caption_labels_with_mask = []
        english_tags_list = [[tag.strip() for tag in sentence.split(' |')] for sentence in english_tags]
        chinese_tags_list = [[tag.strip() for tag in sentence.split(' |')] for sentence in chinese_tags]

        for ind in range(logits.size(0)):
            single_filt_mask = logits[ind].max(dim=1)[0] > self.box_threshold
            single_logits_filt = logits[ind][single_filt_mask]  # num_filt, 256
            single_boxes_filt = boxes[ind][single_filt_mask]  # num_filt, 4

            if len(single_filt_mask) == 0:
                bbox_caption.append('')
                bbox_caption_tokens.append(torch.Tensor([]))
                bbox_caption_tokens_with_mask.append(torch.Tensor([]))
                bbox_caption_labels_with_mask.append(torch.Tensor([]))
                continue

            single_image_bbox_caption = ''
            single_image_bbox_caption_tokens = []
            single_image_bbox_caption_tokens_with_mask = []
            single_image_bbox_caption_labels_with_mask = []
            # get phrase
            tokenized = self.grounding_dino.tokenizer(english_tags[ind])
            # build pred
            pred_phrases = []
            single_image_boxes = []
            single_image_scores = []
            for logit, box in zip(single_logits_filt, single_boxes_filt):
                posmap = logit > self.text_threshold
                assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
                if posmap.dim() == 1:
                    non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
                    # max_idx = posmap.max[1]
                    token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
                    # token_ids = [tokenized["input_ids"][i] for i in non_zero_idx if i == max_idx]
                    pred_phrase = self.grounding_dino.tokenizer.decode(token_ids)
                else:
                    raise NotImplementedError("posmap must be 1-dim")
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                single_image_scores.append(logit.max().item())
                # box = box * torch.Tensor([self.config.image_size, self.config.image_size, self.config.image_size, self.config.image_size])
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]
                single_image_boxes.append(box)
            # print("single_image_boxes:", len(single_image_boxes))
            if len(single_image_boxes) == 0:
                bbox_caption.append('')
                bbox_caption_tokens.append(torch.Tensor([]))
                bbox_caption_tokens_with_mask.append(torch.Tensor([]))
                bbox_caption_labels_with_mask.append(torch.Tensor([]))
                continue

            single_image_boxes = torch.stack(single_image_boxes)
            single_image_scores = torch.Tensor(single_image_scores).to("cuda")
            # nms_idx = torchvision.ops.nms(single_image_boxes, single_image_scores, self.iou_threshold).to('cpu').numpy().tolist()
            nms_idx = torchvision.ops.nms(single_image_boxes, single_image_scores, self.iou_threshold)
            single_image_boxes_filt = single_image_boxes[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]

            # print("single_image_boxes_filt:", single_image_boxes_filt.size())
            # print("pred_phrases:", pred_phrases)
            if single_image_boxes_filt.size(0) == 0:
                bbox_caption.append('')
                bbox_caption_tokens.append(torch.Tensor([]))
                bbox_caption_tokens_with_mask.append(torch.Tensor([]))
                bbox_caption_labels_with_mask.append(torch.Tensor([]))
                continue
            # 处理一条数据的多个框
            for i in range(single_image_boxes_filt.size(0)):
                # ori_box = single_image_boxes_filt[i] / torch.Tensor([self.config.image_size, self.config.image_size, self.config.image_size, self.config.image_size])
                ori_box = single_image_boxes_filt[i]
                # ori_box = torch.Tensor([round(coordinate, 3) for coordinate in single_image_boxes_filt[i]])
                name, _ = pred_phrases[i].split('(')
                name = name.replace('|', '').strip()
                name = re.sub(r'\s-\s', '-', name)
                input_name = None
                # print('english_tags_list:', english_tags_list[ind])
                # print('name:', name)
                if language == 'zh':
                    for tags_ind in range(len(english_tags_list[ind])):
                        if name == english_tags_list[ind][tags_ind]:
                            input_name = chinese_tags_list[ind][tags_ind]
                            break
                    if input_name == None:
                        flag = 0
                        for tags_ind in range(len(english_tags_list[ind])):
                            for name_ind in range(len(name.split()),0,-1):
                                if ' '.join(name.split()[:name_ind]) == english_tags_list[ind][tags_ind]:
                                    input_name = chinese_tags_list[ind][tags_ind]
                                    flag = 1
                                    break
                            for name_ind in range(len(name.split())):
                                if ' '.join(name.split()[name_ind:]) == english_tags_list[ind][tags_ind]:
                                    input_name = chinese_tags_list[ind][tags_ind]
                                    flag = 1
                                    break
                            if flag == 1:
                                break                              
                        if input_name == None:
                            continue                                  
                else:
                    for tags_ind in range(len(english_tags_list[ind])):
                        if name == english_tags_list[ind][tags_ind]:
                            input_name = name
                            break
                    if input_name == None:
                        flag = 0
                        for tags_ind in range(len(english_tags_list[ind])):
                            for name_ind in range(len(name.split()),0,-1):
                                if ' '.join(name.split()[:name_ind]) == english_tags_list[ind][tags_ind]:
                                    input_name = english_tags_list[ind][tags_ind]
                                    flag = 1
                                    break
                            for name_ind in range(len(name.split())):
                                if ' '.join(name.split()[name_ind:]) == english_tags_list[ind][tags_ind]:
                                    input_name = english_tags_list[ind][tags_ind]
                                    flag = 1
                                    break
                            if flag == 1:
                                break
                        if input_name == None:
                            # print('name:', name)
                            # print('input_name:', input_name)
                            # print('english_tags_list:', english_tags_list[ind])
                            # print('input_name is none')
                            input_name = name      
                # if input_name == None:
                #     bbox_caption.append('')
                #     bbox_caption_tokens.append(torch.Tensor([]))
                #     bbox_caption_tokens_with_mask.append(torch.Tensor([]))
                #     bbox_caption_labels_with_mask.append(torch.Tensor([]))                    
                #     continue

                # print('input_name:', input_name)
                single_bbox_caption = input_name + ': [' + ', '.join([str(round(coordinate.item(), 3)) for coordinate in ori_box]) + ']'
                single_image_bbox_caption = single_image_bbox_caption + ' ' + single_bbox_caption
                name_and_bbox_tokens = []
                name_and_bbox_tokens.append(torch.tensor(self.tokenizer(input_name, add_special_tokens=False).input_ids))
                name_and_bbox_tokens.append(torch.tensor(self.tokenizer('[', add_special_tokens=False).input_ids))
                for coordinate in ori_box:
                    name_and_bbox_tokens.append(torch.tensor(self.tokenizer(str(round(coordinate.item(), 3)), add_special_tokens=False).input_ids))
                name_and_bbox_tokens.append(torch.tensor(self.tokenizer(']', add_special_tokens=False).input_ids))

                for name_and_bbox_tokens_ind in range(len(name_and_bbox_tokens)):                     
                    if name_and_bbox_tokens_ind == 1 or name_and_bbox_tokens_ind == 5:
                        single_image_bbox_caption_tokens_with_mask.append(name_and_bbox_tokens[name_and_bbox_tokens_ind])
                        single_image_bbox_caption_labels_with_mask.append(torch.full_like(name_and_bbox_tokens[name_and_bbox_tokens_ind], -100))                           
                    else:
                        if random.random() <= 0.15:
                            single_image_bbox_caption_tokens_with_mask.append(torch.full_like(name_and_bbox_tokens[name_and_bbox_tokens_ind], self.tokenizer.mask_token_id))
                            single_image_bbox_caption_labels_with_mask.append(name_and_bbox_tokens[name_and_bbox_tokens_ind])                            
                        else:
                            single_image_bbox_caption_tokens_with_mask.append(name_and_bbox_tokens[name_and_bbox_tokens_ind])
                            single_image_bbox_caption_labels_with_mask.append(torch.full_like(name_and_bbox_tokens[name_and_bbox_tokens_ind], -100))                     
                    single_image_bbox_caption_tokens.append(name_and_bbox_tokens[name_and_bbox_tokens_ind])
                    
                single_image_bbox_caption_tokens.append(torch.tensor([self.tokenizer.sep_token_id]))
                single_image_bbox_caption_tokens_with_mask.append(torch.tensor([self.tokenizer.sep_token_id]))
                single_image_bbox_caption_labels_with_mask.append(torch.tensor([-100]))
            # try:
            if  single_image_bbox_caption_tokens == '':
                bbox_caption.append('')
                bbox_caption_tokens.append(torch.Tensor([]))
                bbox_caption_tokens_with_mask.append(torch.Tensor([]))
                bbox_caption_labels_with_mask.append(torch.Tensor([]))
                continue 
            single_image_bbox_caption_tokens = torch.cat(single_image_bbox_caption_tokens, dim = -1)
            # except:
            #     print('single_bbox_caption:', single_bbox_caption)
            #     print('single_image_bbox_caption:', single_image_bbox_caption)
            #     print('name_and_bbox_tokens:', name_and_bbox_tokens)
            #     print('single_image_boxes:', single_image_boxes)
            #     print('single_image_boxes_filt:', single_image_boxes_filt)
            #     print('single_image_bbox_caption_tokens:', single_image_bbox_caption_tokens)
            #     print('single_image_boxes_filt:', single_image_boxes_filt.size(1))
            single_image_bbox_caption_tokens_with_mask = torch.cat(single_image_bbox_caption_tokens_with_mask, dim = -1)
            single_image_bbox_caption_labels_with_mask = torch.cat(single_image_bbox_caption_labels_with_mask, dim = -1)
            
            bbox_caption.append(single_image_bbox_caption)
            bbox_caption_tokens.append(single_image_bbox_caption_tokens)
            bbox_caption_tokens_with_mask.append(single_image_bbox_caption_tokens_with_mask)
            bbox_caption_labels_with_mask.append(single_image_bbox_caption_labels_with_mask)
        # if torch.distributed.get_rank() == 0:
        #     print(bbox_caption[0])
        #     print(bbox_caption_tokens[0])
        #     print(self.tokenizer.decode(bbox_caption_tokens[0]))
        #     exit()
        outputs = {'bbox_caption': bbox_caption,
                   'bbox_caption_tokens': bbox_caption_tokens,
                   'bbox_caption_tokens_with_mask': bbox_caption_tokens_with_mask,
                   'bbox_caption_labels_with_mask': bbox_caption_labels_with_mask,
                   }

        return outputs

    def prepare_inputs_for_pretrain(self, captions, bbox_caption_tokens, bbox_caption_tokens_with_mask, bbox_caption_labels_with_mask):
        text_input_tokens_ids = []
        text_input_tokens_ids_with_mask = []
        text_input_labels = []
        text_input_labels_with_mask = []
        text_input_attentions = []
        text_input_attentions_with_mask = []
        text_input_position_ids = []
        text_input_position_ids_with_mask = []
        batch_size = len(captions)         
        for ind, caption in enumerate(captions):
            if len(caption)>500:
                print(caption)
            single_caption_tokens = torch.tensor(self.tokenizer(caption, add_special_tokens=False, truncation=True, max_length=87).input_ids)

            # 单纯caption的label
            single_caption_labels = torch.tensor(single_caption_tokens)
            single_caption_labels_with_mask = torch.full_like(single_caption_tokens, -100)
            
            # 无框
            if bbox_caption_tokens[ind].size(0) == 0:
                single_text_input_tokens_ids = torch.cat([torch.tensor([self.tokenizer.cls_token_id]), single_caption_tokens])
                single_text_input_tokens_ids_with_mask = torch.cat([torch.tensor([self.tokenizer.cls_token_id]), single_caption_tokens])
                text_input_tokens_ids.append(single_text_input_tokens_ids)
                text_input_tokens_ids_with_mask.append(single_text_input_tokens_ids_with_mask)
                text_input_labels.append(torch.cat([torch.tensor([self.tokenizer.bos_token_id]), single_caption_labels]))
                text_input_labels_with_mask.append(torch.cat([torch.tensor([-100]), single_caption_labels_with_mask]))
                text_input_attentions.append(torch.ones_like(single_text_input_tokens_ids))
                text_input_attentions_with_mask.append(torch.ones_like(single_text_input_tokens_ids_with_mask))
                text_input_position_ids.append(torch.cat([torch.Tensor([0]), torch.arange(1, len(single_caption_tokens)+1)]))
                text_input_position_ids_with_mask.append(torch.cat([torch.Tensor([0]), torch.arange(1, len(single_caption_tokens)+1)]))   
                continue
            # 拼接bbox的token和text的token和label
            if len(bbox_caption_tokens[ind]) > self.max_txt_len - 1:
                bbox_caption_tokens[ind] = bbox_caption_tokens[ind][:self.max_txt_len-1]
                bbox_caption_tokens_with_mask[ind] = bbox_caption_tokens_with_mask[ind][:self.max_txt_len-1]
                bbox_caption_labels_with_mask[ind] = bbox_caption_labels_with_mask[ind][:self.max_txt_len-1]

            # LM任务的label
            single_bbox_caption_labels = torch.full_like(bbox_caption_tokens[ind], -100)

            single_text_input_tokens_ids = torch.cat([torch.tensor([self.tokenizer.cls_token_id]), bbox_caption_tokens[ind], single_caption_tokens])
            single_text_input_tokens_ids_with_mask = torch.cat([torch.tensor([self.tokenizer.cls_token_id]), bbox_caption_tokens_with_mask[ind], single_caption_tokens])
            
            #这里也要改 cls 0 其他初始位置1
            single_text_input_position_ids = torch.cat([torch.Tensor([0]), torch.arange(1, len(bbox_caption_tokens[ind])+1), torch.arange(1, len(single_caption_tokens)+1)])
            single_text_input_position_ids_with_mask = torch.cat([torch.Tensor([0]), torch.arange(1, len(bbox_caption_tokens_with_mask[ind])+1), torch.arange(1, len(single_caption_tokens)+1)])                

            single_text_input_labels = torch.cat([torch.tensor([self.tokenizer.bos_token_id]), single_bbox_caption_labels, single_caption_labels])
            single_text_input_labels_with_mask = torch.cat([torch.tensor([-100]), bbox_caption_labels_with_mask[ind], single_caption_labels_with_mask])

            # mask与pad不一样，还是要做注意力
            single_text_input_attentions = torch.ones_like(single_text_input_tokens_ids)
            single_text_input_attentions_with_mask = torch.ones_like(single_text_input_tokens_ids_with_mask)
            # position_ids
            # if single_text_input_tokens_ids.size(-1) > self.max_txt_len:
            #     single_text_input_tokens_ids = single_text_input_tokens_ids[:self.max_txt_len]
            #     single_text_input_tokens_ids_with_mask = single_text_input_tokens_ids_with_mask[:self.max_txt_len]
            #     single_text_input_labels = single_text_input_labels[:self.max_txt_len]
            #     single_text_input_labels_with_mask = single_text_input_labels_with_mask[:self.max_txt_len]
            #     single_text_input_attentions = single_text_input_attentions[:self.max_txt_len]
            #     single_text_input_attentions_with_mask = single_text_input_attentions_with_mask[:self.max_txt_len]
            text_input_tokens_ids.append(single_text_input_tokens_ids)
            text_input_tokens_ids_with_mask.append(single_text_input_tokens_ids_with_mask)
            text_input_labels.append(single_text_input_labels)
            text_input_labels_with_mask.append(single_text_input_labels_with_mask)
            text_input_attentions.append(single_text_input_attentions)
            text_input_attentions_with_mask.append(single_text_input_attentions_with_mask)
            text_input_position_ids.append(single_text_input_position_ids)
            text_input_position_ids_with_mask.append(single_text_input_position_ids_with_mask)   
                
        # 添加一个长度为max_length的tensor
        pad_tensor = torch.ones(self.max_input_len)
        text_input_tokens_ids.append(pad_tensor)
        text_input_tokens_ids_with_mask.append(pad_tensor)
        text_input_labels.append(pad_tensor)
        text_input_labels_with_mask.append(pad_tensor)
        text_input_attentions.append(pad_tensor)
        text_input_attentions_with_mask.append(pad_tensor)
        text_input_position_ids.append(pad_tensor)
        text_input_position_ids_with_mask.append(pad_tensor)

        text_input_tokens_ids = pad_sequence(text_input_tokens_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id)[:batch_size, :self.max_input_len]
        text_input_tokens_ids_with_mask = pad_sequence(text_input_tokens_ids_with_mask, batch_first = True, padding_value = self.tokenizer.pad_token_id)[:batch_size, :self.max_input_len]
        text_input_labels = pad_sequence(text_input_labels, batch_first = True, padding_value = -100)[:batch_size, :self.max_input_len]
        text_input_labels_with_mask = pad_sequence(text_input_labels_with_mask, batch_first = True, padding_value = -100)[:batch_size, :self.max_input_len]
        text_input_attentions = pad_sequence(text_input_attentions, batch_first = True, padding_value = 0)[:batch_size, :self.max_input_len]
        text_input_attentions_with_mask = pad_sequence(text_input_attentions_with_mask, batch_first = True, padding_value = 0)[:batch_size, :self.max_input_len]
        text_input_position_ids = pad_sequence(text_input_position_ids, batch_first = True, padding_value = 0).long()[:batch_size, :self.max_input_len]
        text_input_position_ids_with_mask = pad_sequence(text_input_position_ids_with_mask, batch_first = True, padding_value = 0).long()[:batch_size, :self.max_input_len]   
        
        outputs = {"text_input_tokens_ids": text_input_tokens_ids,
                  "text_input_tokens_ids_with_mask": text_input_tokens_ids_with_mask,
                  "text_input_labels": text_input_labels,
                  "text_input_labels_with_mask": text_input_labels_with_mask,
                  "text_input_attentions": text_input_attentions,
                  "text_input_attentions_with_mask": text_input_attentions_with_mask,
                  "text_input_position_ids": text_input_position_ids,
                  "text_input_position_ids_with_mask": text_input_position_ids_with_mask,
                  }
        return outputs              

    def forward(self, image, grounding_image, ram_image, caption, language):
        image = image

        image_embeds = self.vision_model(image)[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        tags_english, tags_chinese = self.ram.generate_tag(ram_image)

        input_tags = [tag.replace(' |', ',').lower().strip() + "." if not tag.endswith(".") else tag.replace(' |', ',').lower().strip() for tag in tags_english]

        # outputs = self.grounding_dino(grounding_image[None], captions=input_tags)
        outputs = self.grounding_dino(grounding_image, captions=input_tags)
        logits = outputs["pred_logits"].sigmoid()  # (bs, nq, 256)
        boxes = outputs["pred_boxes"]  # (bs, nq, 4)
        detection_image_embeds = outputs["hidden_state"] # (bs, nq, 256)
        detection_image_atts = torch.ones(detection_image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        bbox_outputs = self.generate_bbox_caption(logits, boxes, tags_english, tags_chinese, language)

        text_inputs_for_pretrain = self.prepare_inputs_for_pretrain(caption, bbox_outputs['bbox_caption_tokens'], bbox_outputs['bbox_caption_tokens_with_mask'], bbox_outputs['bbox_caption_labels_with_mask'])   

        # if torch.distributed.get_rank() == 0:
        #     print(self.tokenizer.decode(text_inputs_for_pretrain["text_input_tokens_ids"][0]))
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            detection_encoder_hidden_states=detection_image_embeds,
            detection_encoder_attention_mask=detection_image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_output = self.qformer.bert(
            input_ids=text_inputs_for_pretrain['text_input_tokens_ids'].to('cuda'),
            position_ids=text_inputs_for_pretrain["text_input_position_ids"].to('cuda'),
            attention_mask=text_inputs_for_pretrain['text_input_attentions'].to('cuda'),
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        # print(image_feats.size())
        # print(text_feat.size())
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        # print(text_inputs_for_pretrain['text_input_tokens_ids'].size())
        # print(text_inputs_for_pretrain['text_input_attentions'].size())
        # print(text_inputs_for_pretrain['text_input_position_ids'].size())
        # print(image_embeds)
        # print(detection_image_embeds)
        text_input_ids_world = concat_all_gather(text_inputs_for_pretrain['text_input_tokens_ids'].to('cuda'))
        text_attention_mask_world = concat_all_gather(text_inputs_for_pretrain['text_input_attentions'].to('cuda'))
        text_position_ids_world = concat_all_gather(text_inputs_for_pretrain['text_input_position_ids'].to('cuda'))
        image_embeds_world = all_gather_with_grad(image_embeds)
        detection_image_embeds_world = all_gather_with_grad(detection_image_embeds)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        detection_image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
            detection_image_embeds_neg.append(detection_image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        detection_image_embeds_neg = torch.stack(detection_image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        text_position_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
            text_position_neg.append(text_position_ids_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        text_position_neg = torch.stack(text_position_neg, dim=0)

        text_ids_all = torch.cat(
            [text_inputs_for_pretrain['text_input_tokens_ids'].to('cuda'), text_inputs_for_pretrain['text_input_tokens_ids'].to('cuda'), text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_inputs_for_pretrain['text_input_attentions'].to('cuda'), text_inputs_for_pretrain['text_input_attentions'].to('cuda'), text_atts_neg],
            dim=0,
        )
        position_ids_all = torch.cat(
            [text_inputs_for_pretrain["text_input_position_ids"].to('cuda'), text_inputs_for_pretrain["text_input_position_ids"].to('cuda'), text_position_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        detection_image_embeds_all = torch.cat(
            [detection_image_embeds, detection_image_embeds_neg, detection_image_embeds], dim=0
        )  # pos, neg, pos
        detection_image_atts_all = torch.ones(detection_image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )        

        output_itm = self.qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            position_ids=position_ids_all,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            detection_encoder_hidden_states=detection_image_embeds_all,
            detection_encoder_attention_mask=detection_image_atts_all,            
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_inputs_for_pretrain['text_input_tokens_ids'].to('cuda').clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # labels = decoder_input_ids.masked_fill(
        #     decoder_input_ids == self.tokenizer.pad_token_id, -100
        # )
        print('text_input_tokens_ids:', text_inputs_for_pretrain['text_input_tokens_ids'][0])
        print('text_input_tokens:', self.tokenizer.decode(text_inputs_for_pretrain['text_input_tokens_ids'][0]))
        print('text_input_labels_ids:', text_inputs_for_pretrain['text_input_labels'][0])
        print('text_input_labels:', self.tokenizer.decode(text_inputs_for_pretrain['text_input_labels'][0].masked_fill(text_inputs_for_pretrain['text_input_labels'][0]==-100, torch.tensor(0))))

        decoder_labels = text_inputs_for_pretrain['text_input_labels'].to('cuda')

        decoder_query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        decoder_attention_mask = torch.cat([decoder_query_atts, text_inputs_for_pretrain['text_input_attentions'].to('cuda')], dim=1)
        # print('decoder_input_ids:', decoder_input_ids.size())
        # print('decoder_labels:', decoder_labels.size())
        lm_output = self.qformer(
            decoder_input_ids,
            position_ids=text_inputs_for_pretrain["text_input_position_ids"].to('cuda'),
            attention_mask=decoder_attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=decoder_labels,
            is_decoder=True,
        )

        loss_lm = lm_output.loss

        ##================= Mask Language Model ========================##
        # encoder_input_ids = text_inputs_for_pretrain['text_input_tokens_ids_with_mask'].clone()
        encoder_input_ids = text_inputs_for_pretrain['text_input_tokens_ids_with_mask'].to("cuda")
        # encoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        print('text_input_tokens_ids_with_mask:', text_inputs_for_pretrain['text_input_tokens_ids_with_mask'][0])
        print('text_input_tokens_with_mask:', self.tokenizer.decode(text_inputs_for_pretrain['text_input_tokens_ids_with_mask'][0]))
        print('text_input_labels_ids_with_mask:', text_inputs_for_pretrain['text_input_labels_with_mask'][0])
        print('text_input_labels_with_mask:', self.tokenizer.decode(text_inputs_for_pretrain['text_input_labels_with_mask'][0].masked_fill(text_inputs_for_pretrain['text_input_labels_with_mask'][0]==-100, torch.tensor(0))))

        encoder_labels = text_inputs_for_pretrain['text_input_labels_with_mask'].to("cuda")

        encoder_query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        encoder_attention_mask = torch.cat([encoder_query_atts, text_inputs_for_pretrain['text_input_attentions_with_mask'].to("cuda")], dim=1)
        mlm_output = self.qformer(
            encoder_input_ids,
            position_ids=text_inputs_for_pretrain["text_input_position_ids_with_mask"].to('cuda'),
            attention_mask=encoder_attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=encoder_labels,
            is_decoder=False,
        )

        loss_mlm = mlm_output.loss
        # print(loss_itc)   
        # print(loss_itm)   
        # print(loss_lm)   
        # print(loss_mlm)
        # print('mlm_label:', torch.sum(torch.where(text_inputs_for_pretrain['text_input_labels_with_mask']!=-100,1,0),dim=1))
        # print('lm_label:', torch.sum(torch.where(text_inputs_for_pretrain['text_input_labels']!=-100,1,0),dim=1))     

        return LyricsOutput(
            loss=loss_itc + loss_itm + loss_lm + loss_mlm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_mlm=loss_mlm,
        )


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

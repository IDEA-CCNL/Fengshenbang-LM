# coding=utf-8
# Copyright 2022 IDEA-CCNL The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Della model. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from fengshen.models.deepVAE.configuration_della import DellaModelConfig
from fengshen.models.deepVAE.latent_connector import GPT2ForDecoderLatentConnector, GPT2ForEncoderLatentConnector
from fengshen.models.deepVAE.utils import connect, compute_kl_loss, top_k_top_p_filtering, enforce_repetition_penalty


_CHECKPOINT_FOR_DOC = "della-226M-base"
_CONFIG_FOR_DOC = "DellaModelConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"
Della_model_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "della-226M-base"
]


@dataclass
class DellaModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    posterior_latents: Optional[Tuple[torch.FloatTensor]] = None
    prior_latent: Optional[Tuple[torch.FloatTensor]] = None


class latent_layer(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.W_hh = nn.Linear(input_dim, input_dim, bias=False)
        self.W_ih = nn.Linear(input_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, z_lt_lm1, z_lm1):
        # inputs are z_<l-1 and z_l-1
        return self.tanh(self.W_hh(z_lt_lm1) + self.W_ih(z_lm1))


class AverageSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(hidden_dim)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = torch.tanh

    def forward(self, inputs, attention_mask=None):
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        if attention_mask is not None:
            scores = scores + attention_mask

        scores = self.softmax(scores)
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze(1)

        return representations, scores


class DeepVAE(nn.Module):
    """DeepVAE with recursive latent z extracted from every layer of encoder and applied on every layer of decoder """

    def __init__(self, encoder, decoder, latent_dim, hidden_dim, layer_num, pad_token_id, bos_token_id, eos_token_id, CVAE):
        super(DeepVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.latent_dim = latent_dim
        self.layer_num = layer_num
        self.CVAE = CVAE
        # the first layer of latent net depends on zero vectors and therefore can be ignored
        self.latent_nets = nn.ModuleList([latent_layer(latent_dim) for _ in range(layer_num-1)])
        post_input_dim = hidden_dim+latent_dim if not CVAE else 2*hidden_dim+latent_dim
        prior_input_dim = latent_dim if not CVAE else hidden_dim+latent_dim
        self.posterior_nets = nn.ModuleList([nn.Linear(post_input_dim, 2*latent_dim, bias=False) for _ in range(layer_num)])
        self.prior_nets = nn.ModuleList([nn.Linear(prior_input_dim, 2*latent_dim, bias=False) for _ in range(layer_num)])
        # pooling because we are not using hidden states of BOS token
        self.pooling = nn.ModuleList([AverageSelfAttention(hidden_dim) for _ in range(layer_num)])

    def get_decoder_loss(self, inputs, layer_latent_vecs, cond_inputs):
        loss_mask = None
        dec_inputs = inputs
        if self.CVAE:
            loss_mask = torch.concat((torch.zeros_like(cond_inputs), torch.ones_like(inputs)), dim=1)
            dec_inputs = torch.concat((cond_inputs, inputs), dim=1)
        rec_loss = self.decoder(input_ids=dec_inputs, layer_latent_vecs=layer_latent_vecs,
                                labels=dec_inputs, label_ignore=self.pad_token_id, loss_mask=loss_mask).loss
        rec_loss = rec_loss / torch.sum(inputs != self.pad_token_id, dim=1)  # ignore both the pad token id and the cond inputs
        return rec_loss.mean()

    def get_latent_vecs(self, layer_hidden_states, sample=True, beta_logvar=1., cond_inputs=None):
        prior_z_list, posterior_z_list = [], []
        prior_output_list, posterior_output_list = [], []
        batch_size = layer_hidden_states[0].shape[0]
        z = torch.zeros((batch_size, self.latent_dim), dtype=layer_hidden_states[0].dtype, device=layer_hidden_states[0].device)
        for layer_idx in range(self.layer_num):
            # TODO be more specific about the pooling range, ignore the pad_token_ids could improve the repr of sent or cond inputs
            if self.CVAE:
                cond_length = cond_inputs.shape[-1]
                cond_repr, _ = self.pooling[layer_idx](layer_hidden_states[layer_idx][:, :cond_length, :])
                sent_repr, _ = self.pooling[layer_idx](layer_hidden_states[layer_idx][:, cond_length:, :])
                prior_input = torch.cat([cond_repr, z], dim=1)
                posterior_input = torch.cat([cond_repr, sent_repr, z], dim=1)
            else:
                sent_repr, _ = self.pooling[layer_idx](layer_hidden_states[layer_idx])
                prior_input = z
                posterior_input = torch.cat([sent_repr, z], dim=1)

            prior_net_output = self.prior_nets[layer_idx](prior_input)
            posterior_net_output = self.posterior_nets[layer_idx](posterior_input).squeeze(dim=1)
            prior_z = connect(mean=prior_net_output[:, :self.latent_dim], logvar=prior_net_output[:, self.latent_dim:], sample=sample)
            posterior_z = connect(mean=posterior_net_output[:, :self.latent_dim], logvar=posterior_net_output[:, self.latent_dim:],
                                  sample=sample, beta_logvar=beta_logvar)
            if layer_idx != self.layer_num - 1:
                z = self.latent_nets[layer_idx](z, posterior_z)  # we skip than last iteration
            # save the outputs for decoder and kl loss calculations
            prior_z_list.append(prior_z)
            posterior_z_list.append(posterior_z)
            prior_output_list.append(prior_net_output)
            posterior_output_list.append(posterior_net_output)
        return prior_z_list, posterior_z_list, prior_output_list, posterior_output_list

    def get_kl_loss(self, prior_output_list, posterior_output_list, beta_kl_constraints):
        total_kl_loss = None
        layer_kl_loss = []
        for prior_output, posterior_output in zip(prior_output_list, posterior_output_list):
            kl_loss = compute_kl_loss(posterior_output[:, :self.latent_dim], posterior_output[:, self.latent_dim:],
                                      prior_output[:, :self.latent_dim], prior_output[:, self.latent_dim:])
            # incase of overflow and nan value we shall clip the loss here
            # kl_loss = torch.clip(kl_loss, max=1e4)
            total_kl_loss = kl_loss if total_kl_loss is None else total_kl_loss+kl_loss
            layer_kl_loss.append(kl_loss)
        return total_kl_loss.mean() * beta_kl_constraints, layer_kl_loss

    def forward(self, inputs, beta_kl_constraints, cond_inputs=None):
        # handle cond_inputs differently
        enc_inputs = torch.concat((cond_inputs, inputs), dim=1) if self.CVAE else inputs
        encoder_outputs = self.encoder(input_ids=enc_inputs)
        # hidden_states are tuples with length layer_num+1 and each tensor has shape (batch_size, sequence_length, hidden_size), embedding layer is ignored
        prior_z_list, posterior_z_list, prior_output_list, posterior_output_list = self.get_latent_vecs(
            encoder_outputs.hidden_states[1:], cond_inputs=cond_inputs)
        total_kl_loss, layer_kl_loss = self.get_kl_loss(prior_output_list, posterior_output_list, beta_kl_constraints)
        # pass the posterior to decoder for layer-wise low rank tensor product
        rec_loss = self.get_decoder_loss(inputs, posterior_z_list, cond_inputs)
        return total_kl_loss+rec_loss, rec_loss, total_kl_loss, layer_kl_loss

    def get_cond_prior_vecs(self, layer_hidden_states, cond_inputs, sample=True, beta_logvar=1.):
        prior_z_list, prior_output_list = [], []
        batch_size = layer_hidden_states[0].shape[0]
        z = torch.zeros((batch_size, self.latent_dim), dtype=layer_hidden_states[0].dtype, device=layer_hidden_states[0].device)
        for layer_idx in range(self.layer_num):
            # TODO be more specific about the pooling range, ignore the pad_token_ids could improve the repr of sent or cond inputs
            cond_length = cond_inputs.shape[-1]
            cond_repr, _ = self.pooling[layer_idx](layer_hidden_states[layer_idx][:, :cond_length, :])
            prior_input = torch.cat([cond_repr, z], dim=1)
            prior_net_output = self.prior_nets[layer_idx](prior_input)
            prior_z = connect(mean=prior_net_output[:, :self.latent_dim], logvar=prior_net_output[:, self.latent_dim:],
                              sample=sample, beta_logvar=beta_logvar)
            if layer_idx != self.layer_num - 1:
                z = self.latent_nets[layer_idx](z, prior_z)  # we skip than last iteration
            # save the outputs for decoder and kl loss calculations
            prior_z_list.append(prior_z)
            prior_output_list.append(prior_net_output)
        return prior_z_list, prior_output_list

    def inference(self, inputs, top_p, max_length, top_k=0., temperature=1., repetition_penalty=1., sample=False, beta_logvar=1.):
        # NOTE: if we want to use BOS hidden states for x repr then we need to change the causal mask in attention block.
        encoder_outputs = self.encoder(input_ids=inputs)
        # hidden_states are tuples with length layer_num+1 and each tensor has shape (batch_size, sequence_length, hidden_size), embedding layer is ignored
        if self.CVAE:
            prior_z_list, prior_output_list = self.get_cond_prior_vecs(encoder_outputs.hidden_states[1:], inputs, sample=sample, beta_logvar=beta_logvar)
            latent_vecs = prior_z_list
            generated = inputs
        else:
            prior_z_list, posterior_z_list, prior_output_list, posterior_output_list = self.get_latent_vecs(encoder_outputs.hidden_states[1:], sample=sample, beta_logvar=beta_logvar)
            latent_vecs = posterior_z_list
            generated = [[self.bos_token_id] for _ in range(inputs.shape[0])]
            generated = torch.tensor(generated, dtype=torch.long, device=inputs.device)
        # start generation
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.decoder(input_ids=generated, layer_latent_vecs=latent_vecs, labels=None,
                                       label_ignore=self.pad_token_id)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p, top_k=top_k)
                log_probs = F.softmax(filtered_logits, dim=-1)
                if repetition_penalty != 1.0:
                    enforce_repetition_penalty(log_probs, generated, repetition_penalty)
                next_token = torch.multinomial(log_probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                if all(next_token[idx, 0].item() == self.eos_token_id for idx in range(next_token.shape[0])):
                    break  # if all samples predict eos in the batch.
        return generated


class DellaPretrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """ Initialize the weights """
        pass  # to bypass the not implement error


class Della(DellaPretrainedModel):
    '''This class is only implemented to suit huggingface interface, use vae_pl_module to initialize the VAE for training'''
    config_class = DellaModelConfig
    base_model_prefix = "della"
    supports_gradient_checkpointing = True

    def __init__(self, config: DellaModelConfig):
        super().__init__(config)
        self.config = config
        encoder_model = GPT2ForEncoderLatentConnector(config=self.config)
        decoder_model = GPT2ForDecoderLatentConnector(config=self.config, latent_dim=self.config.latent_dim)
        vae_model = DeepVAE(encoder_model, decoder_model, latent_dim=self.config.latent_dim,
                            hidden_dim=self.config.hidden_size, layer_num=self.config.num_hidden_layers,
                            pad_token_id=self.config.pad_token_id, bos_token_id=self.config.bos_token_id,
                            eos_token_id=self.config.eos_token_id, CVAE=self.config.CVAE)
        self.model = vae_model

    def forward(self, inputs, cond_inputs=None, sample_latent=True):
        # handle cond_inputs differently
        enc_inputs = torch.concat((cond_inputs, inputs), dim=1) if self.model.CVAE else inputs
        encoder_outputs = self.model.encoder(input_ids=enc_inputs)
        # hidden_states are tuples with length layer_num+1 and each tensor has shape (batch_size, sequence_length, hidden_size), embedding layer is ignored
        prior_z_list, posterior_z_list, prior_output_list, posterior_output_list = self.model.get_latent_vecs(
            encoder_outputs.hidden_states[1:], cond_inputs=cond_inputs, sample=sample_latent)

        loss_mask, dec_inputs = None, inputs
        if self.model.CVAE:
            loss_mask = torch.concat((torch.zeros_like(cond_inputs), torch.ones_like(inputs)), dim=1)
            dec_inputs = torch.concat((cond_inputs, inputs), dim=1)
        logits = self.model.decoder(input_ids=dec_inputs, layer_latent_vecs=posterior_z_list,
                                    labels=dec_inputs, label_ignore=self.model.pad_token_id, loss_mask=loss_mask).logits

        return DellaModelOutput(
            logits=logits,
            posterior_latents=posterior_z_list,
            prior_latent=prior_z_list
        )

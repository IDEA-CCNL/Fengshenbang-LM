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

import os
import torch
import numpy as np
from fengshen.models.deepVAE.deep_vae import DeepVAE
from pytorch_lightning.core.lightning import LightningModule
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.bert.tokenization_bert import BertTokenizer
from fengshen.models.deepVAE.latent_connector import GPT2ForDecoderLatentConnector, GPT2ForEncoderLatentConnector
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class DeepVAEModule(LightningModule):
    @classmethod
    def add_module_specific_args(cls, parser):
        group = parser.add_argument_group('vae', 'configurations')
        group.add_argument("--checkpoint_path", type=str, default=None)
        group.add_argument("--gpt2_model_path", type=str)
        group.add_argument("--beta_kl_constraints_start", default=1, type=float,
                           help="min beta for all the latent z posterior vs prior kl loss")
        group.add_argument("--beta_kl_constraints_stop", default=1, type=float,
                           help="max beta for all the latent z posterior vs prior kl loss")
        group.add_argument("--beta_n_cycles", default=30, type=int,
                           help="number of cycles for kl loss ratio within an epoch")
        group.add_argument("--freebit_kl_constraints", default=.1, type=float,
                           help="free bit for all the latent z kl loss")
        group.add_argument("--latent_dim", default=256, type=int,
                           help="latent dimension of deepVAE Z")
        group.add_argument("--learning_rate", default=5e-5, type=float,
                           help="The initial learning rate for Adam.")
        group.add_argument("--weight_decay", default=0.0, type=float,
                           help="Weight deay if we apply some.")
        group.add_argument("--adam_epsilon", default=1e-8, type=float,
                           help="Epsilon for Adam optimizer.")
        group.add_argument("--max_grad_norm", default=1.0, type=float,
                           help="Max gradient norm.")
        group.add_argument("--warmup_steps", default=0, type=int,
                           help="Linear warmup over warmup_steps.")
        group.add_argument("--CVAE", action='store_true',
                           help="specify this argument if finetuning CVAE, otherwise ignore this argument")

        return parser

    @classmethod
    def load_model(cls, args, labels_dict=None):
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'mp_rank_00_model_states.pt'))

        latent_dim = checkpoint['latent_dim'] if ('latent_dim' in checkpoint.keys()) else args.latent_dim
        labels_dict = checkpoint['label_dict'] if ('label_dict' in checkpoint.keys()) else labels_dict

        enc_config = GPT2Config.from_pretrained(args.gpt2_model_path)
        tokenizer = BertTokenizer.from_pretrained(args.gpt2_model_path)
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        # special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'additional_special_tokens': ['<ENT>', '<ENS>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        encoder_model = GPT2ForEncoderLatentConnector(config=enc_config)
        encoder_model.resize_token_embeddings(len(tokenizer))

        dec_config = GPT2Config.from_pretrained(args.gpt2_model_path)
        decoder_model = GPT2ForDecoderLatentConnector(config=dec_config, latent_dim=latent_dim)
        decoder_model.resize_token_embeddings(len(tokenizer))

        vae_model = DeepVAE(encoder_model, decoder_model, latent_dim=latent_dim,
                            hidden_dim=enc_config.hidden_size, layer_num=enc_config.num_hidden_layers,
                            pad_token_id=tokenizer.pad_token_id, unk_token_id=tokenizer.unk_token_id,
                            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                            CVAE=args.CVAE)

        # TODO: all the related params should be loaded here! Including latent_nets, posterior_nets, prior_nets, pooling, decoder.transformer.Wv, decoder.transformer.Wz
        anchor = 'module.model.'
        start = len(anchor)
        vae_dict = {key[start:]: val for key, val in checkpoint['module'].items() if anchor in key}
        # comment out if not initialized from VAE
        # if args.CVAE:
        #     # manually load prior and posterior if initialize CVAE model for the first time because of dim mismatch
        #     prior_post_dict = {key: vae_dict.pop(key) for key in list(vae_dict) if ('posterior_nets' in key or 'prior_nets' in key)}
        #     for idx in range(enc_config.num_hidden_layers):
        #         vae_model.posterior_nets[idx].weight.data[:, enc_config.hidden_size:] = prior_post_dict[f"posterior_nets.{idx}.weight"]
        #         vae_model.prior_nets[idx].weight.data[:, enc_config.hidden_size:] = prior_post_dict[f"prior_nets.{idx}.weight"]
        #     enc_wte_shape, dec_wte_shape  = vae_dict['encoder.transformer.wte.weight'].shape[0], vae_dict['decoder.transformer.wte.weight'].shape[0]
        #     vae_model.encoder.transformer.wte.weight.data[:enc_wte_shape, :] = vae_dict.pop('encoder.transformer.wte.weight')
        #     vae_model.decoder.transformer.wte.weight.data[:dec_wte_shape, :] = vae_dict.pop('decoder.transformer.wte.weight')
        #     vae_model.decoder.lm_head.weight.data[:dec_wte_shape, :] = vae_dict.pop('decoder.lm_head.weight')
        missing_keys, unexpected_keys = vae_model.load_state_dict(vae_dict, strict=False)
        print(f"Vae model loading process: missing keys {missing_keys}, unexpected keys {unexpected_keys}")

        return vae_model, tokenizer

    def __init__(
        self,
        args,
        train_steps=0,
        labels_dict=None
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args

        if args.checkpoint_path is not None:
            self.model, self.encoder_tokenizer, self.decoder_tokenizer, self.latent_dim, \
                self.labels_dict, self.args = DeepVAEModule.load_model(self.args, labels_dict=labels_dict)
        else:
            self.encoder_tokenizer = BertTokenizer.from_pretrained(self.args.encoder_model_path)
            encoder_config = GPT2Config.from_pretrained(self.args.encoder_model_path)
            special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'additional_special_tokens': ['<ENT>', '<ENS>']}
            self.encoder_tokenizer.add_special_tokens(special_tokens_dict)
            self.latent_dim = self.args.latent_dim
            encoder = GPT2ForEncoderLatentConnector.from_pretrained(self.args.encoder_model_path, config=encoder_config)
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
            encoder.resize_token_embeddings(len(self.encoder_tokenizer))

            self.decoder_tokenizer = BertTokenizer.from_pretrained(self.args.decoder_model_path)
            self.decoder_tokenizer.add_special_tokens(special_tokens_dict)
            decoder_config = GPT2Config.from_pretrained(self.args.decoder_model_path)
            self.labels_dict = labels_dict
            decoder = GPT2ForDecoderLatentConnector.from_pretrained(self.args.decoder_model_path, config=decoder_config,
                                                                    latent_dim=self.latent_dim)

            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
            decoder.resize_token_embeddings(len(self.decoder_tokenizer))
            self.model = DeepVAE(encoder, decoder, latent_dim=self.args.latent_dim,
                                 hidden_dim=encoder_config.hidden_size, layer_num=encoder_config.num_hidden_layers,
                                 pad_token_id=self.decoder_tokenizer.pad_token_id, unk_token_id=self.decoder_tokenizer.unk_token_id,
                                 bos_token_id=self.decoder_tokenizer.bos_token_id, eos_token_id=self.decoder_tokenizer.eos_token_id,
                                 CVAE=args.CVAE)

        self.train_steps = train_steps
        # TODO: adjust the cyclic schedule
        self.beta_kl_constraints_list = self.get_cyclic_linear_beta_list(self.train_steps,
                                                                         start=args.beta_kl_constraints_start, stop=args.beta_kl_constraints_stop,  n_cycle=args.beta_n_cycles)
        # self.mlm_probability_list = self.get_decoder_beta_list(self.train_steps,
        #     start=0., stop=1.,  n_cycle=args.beta_n_cycles)
        # self.beta_kl_constraints_list = self.get_constant_ratio(self.train_steps, args.beta_kl_constraints)
        self.mlm_probability_list = self.get_constant_ratio(self.train_steps, 0.)
        # self.freebit_kl_constraints = args.freebit_kl_constraints

    def get_constant_ratio(self, n_steps, ratio):
        L = np.ones(n_steps)
        L *= ratio
        return L

    def get_decoder_beta_list(self, n_steps, start=0., stop=1.0, n_cycle=4):
        L = np.ones(n_steps)
        t_range = int(n_steps / n_cycle)
        for t_cur in range(n_steps):
            if t_cur > t_range:
                L[t_cur] = 0.
            else:
                ratio = t_cur / t_range
                value = stop - ratio * (stop-start)
                L[t_cur] = value
        return L

    def get_cyclic_linear_beta_list(self, n_steps, start=0.5, stop=1.0, n_cycle=4):
        L = np.ones(n_steps)
        t_range = int(n_steps / n_cycle)
        for t_cur in range(n_steps):
            loc = t_cur % t_range
            split_range = int(t_range * 0.25)
            if loc <= 2*split_range:
                value = start
            elif loc <= 3*split_range:
                ratio = (loc % split_range) / split_range
                value = ratio * (stop-start)
            else:
                value = stop
            L[t_cur] = value
        return L

    #####
    # Torch lightning
    #####

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['label_dict'] = self.labels_dict
        checkpoint['latent_dim'] = self.latent_dim

    def training_step(self, batch, batch_idx):
        if batch is None:
            loss = torch.Tensor([0.]).to(next(self.model.parameters()).device)
            loss.requires_grad = True
            return loss
        inputs, cond_inputs = batch, None
        if self.args.CVAE:
            inputs, cond_inputs = batch

        total_loss, rec_loss, total_kl_loss, layer_kl_loss = \
            self.model(inputs, self.beta_kl_constraints_list[batch_idx], cond_inputs)
        # the logging interval are set by the trainer_args log_every_n_steps
        for idx, pg in enumerate(self.optimizers().param_groups):
            self.log(f"learning_rate_{idx}", pg['lr'])
        unscaled_kl_constraint_loss = 0. if self.beta_kl_constraints_list[batch_idx] == 0. else total_kl_loss/self.beta_kl_constraints_list[batch_idx]
        self.log("total_loss", total_loss)
        self.log("total_kl_constraint_loss", total_kl_loss)
        self.log("unscaled_kl_constraint_loss", unscaled_kl_constraint_loss)
        self.log("beta_kl_constraints", self.beta_kl_constraints_list[batch_idx])
        self.log("beta_mlm_probability", self.mlm_probability_list[batch_idx])
        self.log("rec_loss", rec_loss)
        for idx, kl_loss in enumerate(layer_kl_loss):
            self.log(f"layer_{idx}_kl_loss", kl_loss.mean())

        return total_loss

    def training_step_end(self, batch_parts):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        if batch is None:
            loss = torch.Tensor([0.]).to(next(self.model.parameters()).device)
            loss.requires_grad = True
            return loss
        inputs, cond_inputs = batch, None
        if self.args.CVAE:
            inputs, cond_inputs = batch

        total_loss, rec_loss, total_kl_loss, layer_kl_loss = self.model(inputs, 1., cond_inputs)
        # the logging interval are set by the trainer_args log_every_n_steps
        self.log("val_total_loss", total_loss)
        self.log("val_kl_constraint_loss", total_kl_loss)
        self.log("val_recon_loss", rec_loss)
        for idx, kl_loss in enumerate(layer_kl_loss):
            self.log(f"layer_{idx}_kl_loss", kl_loss.mean())
        return total_loss

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        if batch is None:
            loss = torch.Tensor([0.]).to(next(self.model.parameters()).device)
            loss.requires_grad = True
            return loss
        inputs, cond_inputs = batch, None
        if self.args.CVAE:
            inputs, cond_inputs = batch
        total_loss, rec_loss, total_kl_loss, layer_kl_loss = self.model(inputs, 1., cond_inputs)
        self.log("test_total_loss", total_loss)
        self.log("test_recon_loss", rec_loss)
        self.log("test_kl_constraint_loss", total_kl_loss)
        for idx, kl_loss in enumerate(layer_kl_loss):
            self.log(f"layer_{idx}_kl_loss", kl_loss.mean())
        return total_loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.train_steps)

        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
                }

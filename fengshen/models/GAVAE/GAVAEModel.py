# -*- encoding: utf-8 -*-
'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   GAVAEModel.py
@Time    :   2022/11/04 11:35
@Author  :   Liang Yuxin
@Version :   1.0
@Contact :   liangyuxin@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from fengshen.models.DAVAE.DAVAEModel import DAVAEModel
from fengshen.models.GAVAE.gans_model import gans_process


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GAVAEPretrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """ Initialize the weights """
        pass  # to bypass the not implement error

class GAVAEModel(GAVAEPretrainedModel):
    config_class = PretrainedConfig
    def __init__(self, config:PretrainedConfig) -> None:
        super().__init__(config)
        self.config =config
        config.device = device
        self.gan = gans_process(self.config)
        self.vae_model = DAVAEModel(self.config)

    def train_gan(self,encoder_tokenizer,decoder_tokenizer,input_texts):
        self.vae_model.set_tokenizers(encoder_tokenizer,decoder_tokenizer)
        n = len(input_texts)
        inputs_latents = self.vae_model.latent_code_from_text_batch(input_texts)
        well_trained_gan = False
        while not well_trained_gan:
            self.gan_training(inputs_latents)
            latent = torch.tensor(self.gan.gen_test(n))
            if not latent.isnan().any():
                well_trained_gan = True

    def generate(self,n):
        latent_z = torch.tensor(self.gan.gen_test(n)).to(device)
        text = self.vae_model.text_from_latent_code_batch(latent_z,prompt=None)
        return text
    
    def gan_training(self,inputs_latents):
        for gt in range(self.config.gan_epoch):
            x_train,y_train,x_test,y_test,perm = self.gan.ready_cls(inputs_latents)
            # sent_output:latent_z inputs_labels:id of class label
            self.gan.cls_train(x_train, y_train)
            x2_gen, y_gen, s_gen = self.gan.ready_gen(inputs_latents)
            # s_gen:sent_output
            self.gan.gen_train(x2_gen, y_gen, s_gen, gt)

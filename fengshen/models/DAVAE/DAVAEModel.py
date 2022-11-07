import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig,TransfoXLConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import cached_path,hf_bucket_url
from fengshen.models.DAVAE.GPT2ModelForLatent import GPT2ModelForLatent
from fengshen.models.DAVAE.BertForLatentConnector import BertForLatentConnector
from fengshen.models.DAVAE.run_latent_generation import *
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)

class VAEPretrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """ Initialize the weights """
        pass  # to bypass the not implement error

class DAVAEModel(VAEPretrainedModel):
    config_class = PretrainedConfig
    def __init__(self, config:PretrainedConfig,*model_args, **model_kwargs):
        super().__init__(config=config)
        self.config = config
        self.vae_model = DAVAEModel.load_model(self.config)

    @classmethod
    def load_model(cls, config):
        encoder_config = BertConfig.from_dict(config.encoder)
        encoder_model = BertForLatentConnector(config=encoder_config, latent_size=config.latent_size)
        dec_config = TransfoXLConfig.from_dict(config.decoder)
        dec_config.latent_size = config.latent_size
        decoder_model = GPT2ModelForLatent(config=dec_config)
        vae_model = EncDecAAE(config,encoder_model, decoder_model, dec_config.latent_size, pad_token_id=50000)
        return vae_model

    def set_tokenizers(self,encoder_tokenizer,decoder_tokenizer):
        if not hasattr(self, 'encoder_tokenizer'):
            self.encoder_tokenizer = encoder_tokenizer
        if not hasattr(self, 'decoder_tokenizer'):
            self.decoder_tokenizer = decoder_tokenizer
            
    def simulate_batch(self,encoder_tokenizer,decoder_tokenizer, sent_inputs, prompt=None):
        self.set_tokenizers(encoder_tokenizer,decoder_tokenizer)
        # 生成相似句
        latent_z = self.latent_code_from_text_batch(sent_inputs)
        text_analogy = self.text_from_latent_code_batch(latent_z,prompt=prompt)
        return text_analogy
    
    def latent_code_from_text_batch(self,texts):
        # texts->latents
        tokens_tensor_list = []
        for text in texts:
            tokens = self.encoder_tokenizer.encode(text)[:510]
            tokens_tensor_list.append(torch.tensor([101]+tokens+[102]))

        coded = pad_sequence(tokens_tensor_list, batch_first=True, padding_value=0).long()
        device = next(self.vae_model.decoder.parameters()).device
        with torch.no_grad():
            coded = coded.to(device)
            pooled_hidden_fea = self.vae_model.encoder(coded, attention_mask=(coded > 0).float())[1]
            mean, logvar = self.vae_model.encoder.linear(pooled_hidden_fea).chunk(2, -1)

            std = logvar.mul(0.5).exp()
            eps = torch.zeros_like(std).normal_()

            latent_z = mean + torch.mul(eps, std)*self.config.std_scale
            return latent_z
    def text_from_latent_code_batch(self,latent_z, prompt=None):
        # latents->texts
        device = next(self.vae_model.decoder.parameters()).device
        past = latent_z
        batch_size = latent_z.shape[0]
        bos_token = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.bos_token)
        end_token = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.eos_token)

        if prompt is not None:
            prompt = [[bos_token] + self.decoder_tokenizer.encode(text)[:-1] for text in prompt]
        else:
            prompt = [[bos_token]]*batch_size

        context_tokens_tensor = torch.tensor([[end_token]*self.config.max_out_length]*batch_size).to(device) # 2-d tensor
        context_length_tensor = torch.tensor([1]*batch_size).to(device)
        for i in range(batch_size):
            context_tokens_tensor[i,:len(prompt[i])] = torch.tensor(prompt[i]).long().to(device)
            context_length_tensor[i] = len(prompt[i])

        out = sample_sequence_conditional_batch(
            model=self.vae_model.decoder,
            max_out_length= self.config.max_out_length, 
            context_tokens_tensor=context_tokens_tensor,
            context_length_tensor=context_length_tensor,
            latent_z=latent_z,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            device=device
        )

        out_text = []
        for i, tokens in enumerate(out):
            tokens = tokens[len(prompt[i]):]
            tokens = tokens[:tokens.index(end_token)] if end_token in tokens else tokens
            text = self.decoder_tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
            out_text.append(filter_noise(text))
        return out_text
class EncDecAAE(nn.Module):
    """Adversarial Auto-Encoder"""
    def __init__(self,config, encoder, decoder, latent_size, pad_token_id):
        super(EncDecAAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.pad_token_id = pad_token_id
        self.Disc = nn.Sequential(nn.Linear(latent_size, 4*latent_size), nn.ReLU(),
                               nn.Linear(4*latent_size, 1))
        # Standard Normal prior
        loc = torch.zeros(latent_size)
        scale = torch.ones(latent_size)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def connect(self, bert_fea, nsamples=1, fb_mode=0):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        z = self.reparameterize(mean, logvar, nsamples)
        if fb_mode == 0:
            KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        elif fb_mode == 1:
            kl_loss = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
            kl_mask = (kl_loss > self.config.dim_target_kl).float()
            KL = (kl_mask * kl_loss).sum(dim=1)

        return z, KL

    def connect_deterministic(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        logvar = torch.zeros_like(logvar)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device).half()
        ones = torch.ones(len(z), 1, device=z.device).half()

        loss_d = F.binary_cross_entropy_with_logits(self.Disc(z.detach().half()), zeros) + \
        F.binary_cross_entropy_with_logits(self.Disc(zn.half()), ones)
        loss_g = F.binary_cross_entropy_with_logits(self.Disc(z.half()), ones)
        return loss_d, loss_g

    def forward(self, inputs, labels, beta=0.0, iw=None, fb_mode=0, emb_noise=None):
        attention_mask = (inputs > 0).float()
        reconstrution_mask = (labels != self.pad_token_id).float() # the padding token for GPT2
        sent_length = torch.sum(reconstrution_mask, dim=1)

        outputs = self.encoder(inputs, attention_mask, emb_noise=emb_noise)
        pooled_hidden_fea = outputs[1]

        seq_length = labels.size(1)
        dec_attn_mask = self.decoder.get_attn_mask(seq_length).to(labels.device)

        if fb_mode in [0,1]:
            latent_z, loss_kl = self.connect(pooled_hidden_fea, fb_mode=fb_mode)
            latent_z = latent_z.squeeze(1)
            outputs = self.decoder(input_ids=labels, attention_mask=dec_attn_mask, latent_state=latent_z, labels=labels, label_ignore=self.pad_token_id) # ignore loss over padding tokens
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
        elif fb_mode==2: 
            latent_z, loss_kl = self.connect_deterministic(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)
            outputs = self.decoder(input_ids=labels, attention_mask=dec_attn_mask, latent_state=latent_z, labels=labels, label_ignore=self.pad_token_id)
            loss_rec = outputs[0]  # model outputs are always tuple

        if self.config.length_weighted_loss:
            loss = loss_rec / sent_length + beta * loss_kl
        else:
            loss = loss_rec + beta * loss_kl

        if iw!=None:
            total_loss = torch.sum(loss*iw)/torch.sum(iw)
        else:
            total_loss = torch.sum(loss)
        return (loss_rec/sent_length).mean(), loss_kl.mean(), total_loss


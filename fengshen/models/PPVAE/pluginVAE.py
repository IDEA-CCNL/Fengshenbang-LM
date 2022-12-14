import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from fengshen.models.DAVAE.DAVAEModel import DAVAEModel
from fengshen.models.PPVAE.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, latent_dim=128, bottle_dim=20) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim//2)
        self.fc2 = nn.Linear(latent_dim//2, latent_dim//4)
        self.mean = nn.Linear(latent_dim//4, bottle_dim)
        self.log_var = nn.Linear(latent_dim//4, bottle_dim)

    def kl_loss(self, mean, log_var):
        return (-0.5 * (1 + log_var - mean**2 - log_var.exp()).sum(-1)).mean()

    def sampling(self, mean, log_var):
        epsilon = torch.randn(mean.shape[0], mean.shape[-1], device=mean.device)
        return mean + (log_var / 2).exp() * epsilon.unsqueeze(1)

    def forward(self, z):
        '''
        :param z: shape (b, latent_dim)
        '''
        z = self.fc1(z)
        z = F.leaky_relu(z)
        z = F.leaky_relu(self.fc2(z))
        z_mean = self.mean(z)

        z_log_var = self.log_var(z)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        enc_z = self.sampling(z_mean, z_log_var)

        if not self.training:
            enc_z = z_mean
        
        return enc_z, kl_loss

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, bottle_dim=20) -> None:
        super().__init__()
        self.fc1 = nn.Linear(bottle_dim, latent_dim//4)
        self.fc2 = nn.Linear(latent_dim//4, latent_dim//2)
        self.fc3 = nn.Linear(latent_dim//2, latent_dim)

    def forward(self, enc_z):
        z = F.leaky_relu(self.fc1(enc_z))
        z = F.leaky_relu(self.fc2(z))
        z = self.fc3(z)
        return z

class PluginVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.kl_weight = config.kl_weight
        self.beta = config.beta
        self.encoder = Encoder(config.latent_dim, config.bottle_dim)
        self.decoder = Decoder(config.latent_dim, config.bottle_dim)

    def set_beta(self, beta):
        self.beta = beta

    def forward(self, z):
        enc_z, kl_loss = self.encoder(z)
        z_out = self.decoder(enc_z)
        return z_out, kl_loss

    def loss(self, z):
        z_out, kl_loss = self.forward(z)
        z_loss = ((z_out-z)**2).mean()
        loss = z_loss + self.kl_weight * (kl_loss-self.beta).abs()
        return loss, kl_loss

class PPVAEPretrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """ Initialize the weights """
        pass  # to bypass the not implement error

class PPVAEModel(PPVAEPretrainedModel):
    config_class = PretrainedConfig
    def __init__(self, config:PretrainedConfig) -> None:
        super().__init__(config=config)
        self.config =config
        self.pluginvae = PluginVAE(self.config)
        self.vae_model = DAVAEModel(self.config)

    def train_plugin(self,encoder_tokenizer,decoder_tokenizer,input_texts,negative_samples=None):
        # 输入：pluginVAE,label,train_data_dict
        # 输出：pluginVAE
        self.vae_model.set_tokenizers(encoder_tokenizer,decoder_tokenizer)
        pos=self.get_latent(input_texts)
        pos_batch_size = self.config.batch_size
        total_epoch = self.config.total_epoch
        pos_dataset = CustomDataset(pos)
        pos_dataloader = DataLoader(
            pos_dataset,
            batch_size=pos_batch_size,
            shuffle=True
        )
        neg =None
        if negative_samples is not None:
            neg=self.get_latent(negative_samples)
            neg_batch_size = int(pos_batch_size*(neg.shape[0]/pos.shape[0]))
            neg_dataset = CustomDataset(neg)
            neg_dataloader = DataLoader(
                neg_dataset,
                batch_size=neg_batch_size,
                shuffle=True
            )
        optimizer = torch.optim.Adam(
            params=self.pluginvae.parameters(),
            lr=self.config.ppvae_lr, betas=(self.config.mu, self.config.nu)
        )
        gamma = self.config.gamma
        iter_num = 0
        early_stopper = EarlyStopping()
        min_loss = 10.0
        for epoch in range(total_epoch):
            self.pluginvae.train()
            total_pos_loss = 0.0
            total_neg_loss = 0.0
            total_loss = 0.0
            total_pos_kl = 0.0
            for i, data in enumerate(pos_dataloader): 
                if self.config.get_dymanic_beta:
                    self.pluginvae.set_beta(self.get_beta_weight(iter_num,self.config.beta,self.config.beta_total_step))
                iter_num += 1
                pos_loss,pos_kl = self.pluginvae.loss(data)
                neg_loss = 0.0
                if neg is not None:
                    neg_data = next(iter(neg_dataloader))
                    neg_loss,loss_kl = self.pluginvae.loss(neg_data)
                    if neg_loss.item()>self.config.neg_loss_threshold*pos_loss.item():
                        # print("neg_loss exceed, detached")
                        neg_loss = neg_loss.detach()
                    total_neg_loss += neg_loss.item()
                loss = pos_loss - gamma*neg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_pos_loss += pos_loss.item()
                total_loss += loss.item()
                total_pos_kl += pos_kl.item()
            avg_loss = total_loss/len(pos_dataloader)
            avg_kl_loss = total_pos_kl/len(pos_dataloader)
            if avg_loss<min_loss:
                min_loss = avg_loss
                early_stopper.counter = 0
            early_stopper(avg_loss, min_loss)
            if early_stopper.early_stop:
                # print(f"stop training at epoch {epoch}")
                break

    def generate(self,n):
        latent_z = self.gen_latent(n)
        text_analogy = self.vae_model.text_from_latent_code_batch(latent_z)
        return text_analogy

    def get_latent(self,texts):
        latent = self.vae_model.latent_code_from_text_batch(texts)
        return latent

    def gen_latent(self,gen_num=5):
        random_vec = torch.randn((gen_num, self.config.bottle_dim)).to(device)
        with torch.no_grad():
            g_vec = self.pluginvae.decoder(random_vec)
        return g_vec

    def get_beta_weight(self,iter_num,beta,total_step):
        now_beta_weight = min((beta/total_step)*iter_num, beta)
        return now_beta_weight


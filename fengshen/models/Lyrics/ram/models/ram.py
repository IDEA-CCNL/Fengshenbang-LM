'''
 * The Recognize Anything Model (RAM) & Tag2Text Model
 * Written by Xinyu Huang
'''
import numpy as np
import json
import torch
import warnings
import sys
from torch import nn
from fengshen.models.groundedblip.ram.models.bert import BertConfig, BertModel, BertLMHeadModel
from fengshen.models.groundedblip.ram.models.swin_transformer import SwinTransformer
from fengshen.models.groundedblip.ram.data.ram_tag_list_threshold import ram_class_threshold

from fengshen.models.groundedblip.ram.models.utils import *

warnings.filterwarnings("ignore")


class RAM(nn.Module):
    def __init__(self,
                 args):
        r""" The Recognize Anything Model (RAM) inference module.
        RAM is a strong image tagging model, which can recognize any common category with high accuracy.
        Described in the paper " Recognize Anything: A Strong Image Tagging Model" https://recognize-anything.github.io/
        
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
            threshold (int): tagging threshold
            delete_tag_index (list): delete some tags that may disturb captioning
        """
        super().__init__()

        # create image encoder

        if args.vit == 'swin_l':
            assert args.image_size == args.image_res
            # assert config['patch_size'] == 32
            vision_width =args.vision_width

            self.visual_encoder = SwinTransformer(
                img_size=args.image_res,
                patch_size=4,
                in_chans=3,
                embed_dim=args.embed_dim,
                depths=args.depths,
                num_heads=args.num_heads,
                window_size=args.window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)
            
        # create tokenzier
        self.tokenizer = init_tokenizer()

        # Tag2Text employ encoder-decoder architecture for image-tag-text generation: image-tag interaction encoder and image-tag-text decoder
        # create image-tag interaction encoder
        encoder_config = BertConfig.from_json_file(args.med_config)
        encoder_config.encoder_width = 512
        self.tag_encoder = BertModel(config=encoder_config,
                                     add_pooling_layer=False)

        # create image-tag-text decoder
        decoder_config = BertConfig.from_json_file(args.med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        self.delete_tag_index = args.delete_tag_index
        self.prompt = args.prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # load tag list
        self.tag_list = self.load_tag_list(args.tag_list)
        self.tag_list_chinese = self.load_tag_list(args.tag_list_chinese)

        # create image-tag recognition decoder
        self.threshold = args.threshold
        self.num_class = len(self.tag_list)
        q2l_config = BertConfig.from_json_file(args.q2l_config)
        q2l_config.encoder_width = 512
        self.tagging_head = BertModel(config=q2l_config,
                                      add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))
        self.label_embed = nn.Embedding(self.num_class, q2l_config.hidden_size)

        if q2l_config.hidden_size != 512:
            self.wordvec_proj = nn.Linear(512, q2l_config.hidden_size)
        else:
            self.wordvec_proj = nn.Identity()

        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()

        # share weights of the lowest 2-layer of "image-tag interaction encoder" with the "image-tag recogntion decoder"
        tie_encoder_decoder_weights(self.tag_encoder, self.tagging_head, '',
                                    ' ')
        self.image_proj = nn.Linear(vision_width, 512)
        # self.label_embed = nn.Parameter(torch.load(args.textual_label_embedding_path,map_location='cpu').float())
        self.label_embed = nn.Parameter(torch.zeros(4585, 512, dtype=float)) # 4585是标签个数
        # adjust thresholds for some tags
        self.class_threshold = torch.ones(self.num_class) * self.threshold
        for key,value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    # delete self-attention layer of image-tag recognition decoder to reduce computation, follower Query2Label
    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention

    def generate_tag(self,
                 image,
                 threshold=0.68,
                 tag_input=None,
                 ):
            
        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        # recognized image tags using image-tag recogntiion decoder
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device))

        tag = targets.cpu().numpy()
        tag[:,self.delete_tag_index] = 0
        tag_output = []
        tag_output_chinese = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(' | '.join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(' | '.join(token_chinese))


        return tag_output, tag_output_chinese

    def generate_tag_zeroshot(self,
                 image,
                 threshold=0.68,
                 tag_input=None,
                 ):
            
        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        # recognized image tags using image-tag recogntiion decoder
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device))

        tag = targets.cpu().numpy()
        tag[:,self.delete_tag_index] = 0
        tag_output = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(' | '.join(token))

        return tag_output


import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import timm
from torchvision import transforms as T
import open_clip
import sys
import torch
import json
from transformers import BertModel, BertTokenizer
from PIL import Image
from diffusers import StableDiffusionPipeline
import random
import os
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING']='1'
torch.backends.cudnn.benchmark = True

class AestheticsMLP(pl.LightningModule):
    # 美学判别器是基于CLIP的基础上接了一个MLP
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class WaterMarkModel(nn.Module):
    def __init__(self, model_path='./watermark_model_v1.pt'):
        super(WaterMarkModel, self).__init__()
        # model definition
        self.model = timm.create_model(
                'efficientnet_b3a', pretrained=True, num_classes=2)

        self.model.classifier = nn.Sequential(
            # 1536 is the orginal in_features
            nn.Linear(in_features=1536, out_features=625),
            nn.ReLU(),  # ReLu to be the activation function
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )
        self.model.load_state_dict(torch.load(model_path))
    def forward(self, x):
        return self.model(x)

class FilterSystem:
    def __init__(
                    self, 
                    clip_model_path="IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese",
                    aesthetics_model_path="./ava+logos-l14-linearMSE.pth",
                    watermark_model_path="./watermark_model_v1.pt"
                ):
        self.clip_model_path = clip_model_path
        self.aesthetics_model_path = aesthetics_model_path
        self.watermark_model_path = watermark_model_path
        self.init_aesthetics_model()
        self.init_clip_model()
        self.init_watermark_model()

    def init_clip_model(self, ):
        # 此处初始化clip模型，返回模型、tokenizer、processor
        text_encoder = BertModel.from_pretrained(self.clip_model_path).eval().cuda()
        text_tokenizer = BertTokenizer.from_pretrained(self.clip_model_path)
        clip_model, _, processor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        clip_model = clip_model.eval().cuda()
        self.text_encoder, self.text_tokenizer, self.clip_model, self.processor = text_encoder, text_tokenizer, clip_model, processor
        print("clip model loaded")
        return None

    def init_aesthetics_model(self, ):
        # 此处初始化美学模型
        self.aesthetics_model = AestheticsMLP(768)
        self.aesthetics_model.load_state_dict(torch.load(self.aesthetics_model_path))
        self.aesthetics_model.eval().cuda()
        print("aesthetics model loaded")
        return None

    def init_watermark_model(self, ):
        self.watermark_model = WaterMarkModel(self.watermark_model_path)
        self.watermark_model.eval().cuda()
        self.watermark_processor =  T.Compose([
                                                T.Resize((256, 256)),
                                                T.ToTensor(),
                                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        print("watermark model loaded")
        return None

    def get_image_feature(self, images):
        # 此处返回图像的特征向量
        if isinstance(images, list):
            images = torch.stack([self.processor(image) for image in images]).cuda()
        elif isinstance(images, torch.Tensor):
            images = images.cuda()
        else:
            images = self.processor(images).cuda()

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=1, keepdim=True)
        return image_features
    
    def get_text_feature(self, text):
        # 此处返回文本的特征向量
        if isinstance(text, list) or isinstance(text, str):
            text = self.text_tokenizer(text, return_tensors='pt', padding=True)['input_ids'].cuda()
        elif isinstance(text, torch.Tensor):
            text = text.cuda()

        with torch.no_grad():
            text_features = self.text_encoder(text)[1]
            text_features /= text_features.norm(dim=1, keepdim=True)
        return text_features

    def calculate_clip_score(self, features1, features2):
        # 此处2个特征向量的相似度，输入可以是 图片+文本、文本+文本、图片+图片。
        # 返回的是相似度矩阵，维度为 f1.shape[0] * f2.shape[0]
        score_matrix =  features1 @ features2.t()
        return score_matrix

    def get_clip_score(self, text, image):
        text_feature = self.get_text_feature(text)
        image_feature = self.get_image_feature(image)
        return self.calculate_clip_score(text_feature, image_feature)

    def get_aesthetics_score(self, features):
        # 此处返回美学分数，传入的是CLIP的feature, 先计算get_image_feature在传入此函数~(模型是ViT-L-14)
        with torch.no_grad():
            scores = self.aesthetics_model(features)
            scores = scores[:, 0].detach().cpu().numpy()
        return scores

    def get_watermark_score(self, images):
        if isinstance(images, list):
            images = torch.stack([self.watermark_processor(image) for image in images]).cuda()
        elif isinstance(images, torch.Tensor):
            images = images.cuda()
        with torch.no_grad():
            pred = self.watermark_model(images)
            watermark_scores = F.softmax(pred, dim=1)[:,0].detach().cpu().numpy()

        return watermark_scores

class InferenceFlickr:
    def __init__(self, sd_model_list, sample_num=20, guidance_scale=7.5, test_caption_path="/cognitive_comp/chenweifeng/project/dataset/mm_data/Flickr30k-CNA/test/flickr30k_cn_test.txt"):
        self.model_name_list = sd_model_list
        self.guidance_scale = guidance_scale
        self.sample_num=sample_num
        self.score_model = FilterSystem()
        self.caption_path = test_caption_path
        self.score = dict()
        self.final_score = dict()

    def init_model(self):
        self.model_list = []
        for model_name in self.model_name_list:
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
            self.model_list.append(pipe)

    def generate_image_score(self, prompt_list, model_list):
        generator = torch.Generator(device=0)
        generator = generator.manual_seed(42)
        # num_images = 1
        # latents = None
        # seeds = []
        # for _ in range(num_images):
        #     generator = generator.manual_seed(42)
            
        #     image_latents = torch.randn(
        #         (1, pipe.unet.in_channels, 512 // 8, 512 // 8),
        #         generator = generator,
        #         device =1
        #     )
        #     latents = image_latents if latents is None else torch.cat((latents, image_latents))
        for i, model in enumerate(model_list):
            model_name = self.model_name_list[i]
            self.score[model_name] = dict()
            for j, prompt in tqdm(enumerate(prompt_list)):
                latents = None
                image_latents = torch.randn(
                    (1, model.unet.in_channels, 512 // 8, 512 // 8),
                    generator = generator,
                    device =0,
                    dtype=torch.float16
                )
                latents = image_latents if latents is None else torch.cat((latents, image_latents))
                image = model(prompt, guidance_scale=self.guidance_scale, latents=latents, torch_dtype=torch.float16).images[0]
                image_feature = self.score_model.get_image_feature([image])
                text_feature = self.score_model.get_text_feature(prompt)
                image_clip_score = self.score_model.calculate_clip_score(image_feature, text_feature)
                image_watermark_score = self.score_model.get_watermark_score([image])
                image_aesthetics_score =self.score_model.get_aesthetics_score(image_feature)
                self.score[model_name][prompt] = {
                    "clip_score": float(image_clip_score[0][0]),
                    "watermark_score": float(image_watermark_score[0]),
                    "aesthetics_score": float(image_aesthetics_score[0]),
                }
                image.save(f"tmp/{prompt}_model-{str(i)}.png")

    def get_prompt_list(self, seed=42, ):
        with open(self.caption_path) as fin:
            input_lines = fin.readlines()
        tmp_list = []
        for line in input_lines:
            infos = line.strip('\n').split('\t')
            prompt = infos[1]
            tmp_list.append(prompt)
        random.seed(seed)
        prompt_list = random.sample(tmp_list, self.sample_num)
        return prompt_list

    def run(self):
        self.init_model()
        prompt_list = self.get_prompt_list()
        self.generate_image_score(prompt_list, self.model_list)
        
    def show(self, save_path=None):
        # print(self.score)
        print(self.final_score)
        if save_path:
            with open(save_path, 'w') as fout:
                json.dump(fout, self.final_score, indent=2, ensure_ascii=False)
    
    def calculate_score(self,):
        for model_name in self.score.keys():
            clip_score = 0.0
            watermark_score = 0.0
            aesthetics_score = 0.0
            for prompt in self.score[model_name]:
                clip_score += self.score[model_name][prompt]['clip_score']
                watermark_score += self.score[model_name][prompt]['watermark_score']
                aesthetics_score += self.score[model_name][prompt]['aesthetics_score']
            average_clip_score = clip_score / len(self.score[model_name].keys())
            average_watermark_score = watermark_score / len(self.score[model_name].keys())
            average_aesthetics_score = aesthetics_score / len(self.score[model_name].keys())
            self.final_score[model_name] = {"avg_clip": average_clip_score, "avg_watermark": average_watermark_score, 'avg_aesthetics': average_aesthetics_score}

def main():
    model_path = sys.argv[1]
    model_list = [
        # '/cognitive_comp/chenweifeng/project/stable-diffusion-lightning/finetune_taiyi_v0.40_laion',
        # '/cognitive_comp/chenweifeng/project/stable-diffusion-chinese/finetune_taiyi0'
        # "/cognitive_comp/lixiayu/diffuser_models/wukong_epoch1"
        # "/cognitive_comp/lixiayu/work/Fengshenbang-LM/fengshen/workspace/taiyi-stablediffusion-laion/60per_ckpt",
        model_path
    ]
    score_model = InferenceFlickr(model_list, sample_num=1000)
    score_model.run()
    score_model.calculate_score()
    score_model.show()

if __name__ == "__main__":
    main()

from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import (
    WandbLogger
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from fengshen.models.lyrics.modeling_lyrics import LyricsLMForConditionalGeneration
import fengshen.models.lyrics.groundingdino.transforms as T
from fengshen.models.lyrics.configuration_lyrics import LyricsConfig
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
import argparse
from peft import LoraConfig, get_peft_config, get_peft_model
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
import numpy as np
from torchvision.transforms import Normalize, Compose, Resize, RandomResizedCrop, InterpolationMode, ToTensor, RandomHorizontalFlip
from PIL import Image
from transformers import BertTokenizer, Blip2Processor, InstructBlipProcessor, InstructBlipForConditionalGeneration, LlamaTokenizer
from torch.utils.data._utils.collate import default_collate
import os
import torch
import random
from io import BytesIO
from base64 import b64decode

# BlipImageProcessor
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

class Collator():
    def __init__(self, args):
        self.processor = InstructBlipProcessor.from_pretrained(args.model_path, padding_side = "right")
        self.eos_token = self.processor.tokenizer.eos_token
        self.grounding_transforms = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.ram_transforms = Compose([
                    Resize((384, 384)),
                    ToTensor(), 
                    Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
                ])
        self.max_txt_len = 24 
        self.max_output_txt_len = 40 
        self.prompts = {
            "zh":[
                "在本任务中，您将获得一张图片，您的任务是生成该图片的描述。",
                "在这项任务中，您将获得一篇图片。你的任务是用一句话概括这张图片。",
                "为给定的图片生成一个适当的描述。",
                "本任务中，您将获得一张图片。你的任务是描述它。",
                "这张图片的内容是什么。",
                "请简单描述一下这张图片。",
            ],
            "en": [
                'A short image caption:',
                'A short image description:',
                'A photo of',
                'An image that shows',
                'Write a short description for the image.',
                'Write a description for the photo.',
                'Provide a description of what is presented in the photo.',
                'Briefly describe the content of the image.',
                'Can you briefly explain what you see in the image?',
                'Could you use a few words to describe what you perceive in the photo?',
                'Please provide a short depiction of the picture.',
                'Using language, provide a short account of the image.',
                'Use a few words to illustrate what is happening in the picture.',
                ]
        }
        self.stage = 'second' # 一阶段和二阶段代表有无instruct部分

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:], 
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def __call__(self, inputs):
        # samples = []
        images = []
        grounding_pixel_values = []
        ram_pixel_values = []
        questions = []
        answers = []

        for (cnt, i) in enumerate(inputs):
            if 'npy_path' in i:
                instance_image = Image.fromarray(np.load(i['npy_path']))
            elif 'img_path' in i:
                try:
                    instance_image = Image.open(i['img_path'])
                    if not instance_image.mode == "RGB":
                        instance_image = instance_image.convert("RGB")
                except:
                    continue
            elif "image" in i and i["image"] is not None:
                instance_image = i["image"]
                if not instance_image.mode == "RGB":
                    instance_image = instance_image.convert("RGB")
            elif "img" in i and i["img"] is not None:
                instance_image = i["img"]
                if not instance_image.mode == "RGB":
                    instance_image = instance_image.convert("RGB")
            elif "image_base64_str" in i:
                try:
                    instance_image = Image.open(BytesIO(b64decode(i["image_base64_str"][0])))
                    if not instance_image.mode == "RGB":
                        instance_image = instance_image.convert("RGB")
                except:
                    continue
            else:
                raise ValueError('no img path in samples')
            if 1 in instance_image.size:
                continue

            if 'caption' in i:
                answer = i['caption'] + ' ' + self.eos_token
                prompts = self.prompts['en']
                prompt = prompts[random.randint(0, len(prompts) - 1)]
            elif 'caption_zh' in i:
                answer = i['caption_zh'] + ' ' + self.eos_token
                prompts = self.prompts['zh']
                prompt = prompts[random.randint(0, len(prompts) - 1)]
            elif 'text' in i:
                answer = i['text'][0]['answer'] + ' ' + self.eos_token
                prompt = i['text'][0]['question']
            elif 'instruction' in i:
                answer = i['outputs'] + ' ' + self.eos_token
                # if random.random() <=0.15:
                prompt = i['instruction'] + i['inputs']               
            # elif 'caption_zh' in i:
            #     caption = i['caption_zh']
            elif 'question' in i:
                answer = i['answer']+ ' ' + self.eos_token
                prompt = i['question']            

            images.append(instance_image)
            grounding_pixel_values.append(self.grounding_transforms(instance_image, None)[0])
            ram_pixel_values.append(self.ram_transforms(instance_image))
            questions.append(prompt)
            answers.append(answer)

        self.processor.tokenizer.truncation_side = "left"
        # print(questions)
        #  
        text_input_tokens = self.processor(text=questions, padding="longest", truncation=True, max_length=self.max_txt_len, return_tensors="pt")
        # print('text_input_tokens:', text_input_tokens.input_ids)
        
        self.processor.tokenizer.truncation_side = 'right'
        text_output_tokens = self.processor(text=answers, padding="longest", truncation=True, max_length=self.max_output_txt_len, return_tensors="pt")
        # print('text_output_tokens:', text_output_tokens.input_ids)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        labels = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.processor.tokenizer.pad_token_id, -100
        )
        for i, l in enumerate(input_part_targets_len):
            labels[i][:l] = -100        

        images_pixel_values = self.processor.image_processor(images=images, return_tensors="pt")
        # images_pixel_values = torch.stack(images_pixel_values['pixel_values'])
        model_inputs = {
            "pixel_values":images_pixel_values['pixel_values'],
            "grounding_pixel_values": grounding_pixel_values,
            "ram_pixel_values": torch.stack(ram_pixel_values),
            "input_ids": llm_tokens['input_ids'],
            "attention_mask": llm_tokens['attention_mask'],
            "qformer_input_ids": text_input_tokens.qformer_input_ids,
            "qformer_attention_mask": text_input_tokens.qformer_attention_mask,
            "labels": labels,
            }        
        return model_inputs


class Lyrics(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Lyrics')
        parser.add_argument('--freeze_image_tower', default=False, action='store_true')
        parser.add_argument('--freeze_qformer', default=False, action='store_true')
        parser.add_argument('--lora-r', type=int, default=8,
                            help='curvature.')
        parser.add_argument('--inference_mode', type=bool, default=False,
                    help='The inference mode.')
        parser.add_argument('--lora-alpha', type=int, default=32,
                            help='The initialization coefficient of lora-alpha.')  
        parser.add_argument('--lora-dropout', type=int, default=0.05,
                            help='The initialization coefficient of lora_dropout.')
        parser.add_argument('--use-lora', action='store_true', help='LORA.')                
        return parent_parser

    def __init__(self, args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(args)

        # self.model = LyricsQFromerForPretrain.from_pretrained(args.model_path, ignore_mismatched_sizes=True)
        self.model = LyricsLMForConditionalGeneration.from_pretrained(args.model_path)
        self.processor = InstructBlipProcessor.from_pretrained(args.model_path, padding_side = "right")

        self.model.box_threshold = 0.25
        self.model.text_threshold = 0.2
        self.model.iou_threshold = 0.6

        if args.freeze_image_tower:
            self.model.vision_model.eval()
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            self.model.ram.eval()
            for param in self.model.ram.parameters():
                param.requires_grad = False
            self.model.grounding_dino.eval()
            for param in self.model.grounding_dino.parameters():
                param.requires_grad = False
        # if args.freeze_qformer:
        #     self.model.qformer.eval()
        #     self.model.qformer.requires_grad_(False)
        #     self.model.query_tokens.requires_grad_(False)
        # freeze lm
        if args.use_lora:
            # for param in self.model.parameters():
            #     # freeze base model's layers
            #     param.requires_grad = False
            peft_config = LoraConfig(
                target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj)', 
                inference_mode=args.inference_mode, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout
            )
            self.model = get_peft_model(self.model, peft_config)
            # self.model.base_model.model.qformer.train()
            # self.model.base_model.model.qformer.requires_grad_(True)
            # self.model.base_model.model.query_tokens.requires_grad_(True)
            # self.model.base_model.model.language_projection.train()
            # self.model.base_model.model.language_projection.requires_grad_(True)
            self.model.print_trainable_parameters()
        elif args.freeze_qformer:
            self.model.language_model.eval()
            self.model.language_model.requires_grad_(False)
            self.model.qformer.eval()
            self.model.qformer.requires_grad_(False)
            self.model.query_tokens.requires_grad_(False)             
        else:
            self.model.language_model.eval()
            self.model.language_model.requires_grad_(False)
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad == True:
        #         print(name)

                                              

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            self.steps_per_epoch = self.total_steps // self.trainer.max_epochs
            print('Total steps: {}' .format(self.total_steps))
        elif stage == 'validate':
            self.total_steps = 100

    def configure_optimizers(self):
        return configure_optimizers(self)
    
    def detokenize(self, token_ids):
        toks = self.processor.tokenizer.convert_ids_to_tokens(token_ids)
        return self.processor.tokenizer.convert_tokens_to_string(toks)
    
    def qformer_detokenize(self, token_ids):
        toks = self.processor.qformer_tokenizer.convert_ids_to_tokens(token_ids)
        return self.processor.qformer_tokenizer.convert_tokens_to_string(toks)    

    def training_step(self, batch):
        if self.trainer.global_rank == 0:
            global SHOW_DATA
            if self.trainer.global_step % 1000 == 0:
                SHOW_DATA = True
                print(f"input_ids: {batch['input_ids'][0]}")
                print(f"input: {self.detokenize(batch['input_ids'][0])}")
                print(f"labels_id: {batch['labels'][0]}")
                print(f"qformer_input_ids: {batch['qformer_input_ids'][0]}")
                print(f"qformer_input: {self.qformer_detokenize(batch['qformer_input_ids'][0])}")

        output = self.model(**batch)
        return output.loss

    def validation_step(self, batch, batch_idx):
        raise Exception("not impl")

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset

    def on_save_checkpoint(self, checkpoint) -> None:
        # 保存的时候把权重按huggingface的形式保存出来
        if self.global_rank == 0:
            dir_path = os.path.join(
                self.hparams.default_root_dir, f'hf_out_{self.trainer.current_epoch}_{self.trainer.global_step}')
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            self.model.save_pretrained(dir_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = add_data_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = Lyrics.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    # wandb_logger = WandbLogger(project="Lyrics")  # 初始化个WandbLogger对象
    trainer = Trainer.from_argparse_args(args,
                                        #  logger=wandb_logger,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    model = Lyrics(args)
    collate_fn = Collator(args)
    datasets = load_data(args, global_rank=trainer.global_rank)
    # print(datasets)
    datamoule = UniversalDataModule(
        tokenizer=None, collate_fn=collate_fn, args=args, datasets=datasets)
    # trainer.fit(model, datamoule)
    trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)

# from utils import *
import subprocess
import io
import torch.nn as nn
from torch.nn import functional as F
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import math
import requests
import cv2
from resize_right import resize
from guided_diffusion.script_util import model_and_diffusion_defaults
from types import SimpleNamespace
from PIL import Image
import argparse
from guided_diffusion.unet import HFUNetModel
from tqdm.notebook import tqdm
from datetime import datetime
from guided_diffusion.script_util import create_model_and_diffusion
import clip
from transformers import BertForSequenceClassification, BertTokenizer
import gc
import random
import os
import sys
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{PROJECT_DIR}/guided-diffusion')   # 加在前面，不再读取库文件的东西。


# ======================== GLOBAL SETTING ========================

useCPU = False  # @param {type:"boolean"}
skip_augs = False  # @param{type: 'boolean'}
perlin_init = False  # @param{type: 'boolean'}

use_secondary_model = False
diffusion_model = "custom"

# Dimensions must by multiples of 64.
side_x = 512
side_y = 512

diffusion_sampling_mode = 'ddim'  # @param ['plms','ddim']
use_checkpoint = True  # @param {type: 'boolean'}
ViTB32 = False  # @param{type:"boolean"}
ViTB16 = False  # @param{type:"boolean"}
ViTL14 = True  # @param{type:"boolean"}
ViTL14_336px = False  # @param{type:"boolean"}
RN101 = False  # @param{type:"boolean"}
RN50 = False  # @param{type:"boolean"}
RN50x4 = False  # @param{type:"boolean"}
RN50x16 = False  # @param{type:"boolean"}
RN50x64 = False  # @param{type:"boolean"}


# @markdown #####**OpenCLIP settings:**
ViTB32_laion2b_e16 = False  # @param{type:"boolean"}
ViTB32_laion400m_e31 = False  # @param{type:"boolean"}
ViTB32_laion400m_32 = False  # @param{type:"boolean"}
ViTB32quickgelu_laion400m_e31 = False  # @param{type:"boolean"}
ViTB32quickgelu_laion400m_e32 = False  # @param{type:"boolean"}
ViTB16_laion400m_e31 = False  # @param{type:"boolean"}
ViTB16_laion400m_e32 = False  # @param{type:"boolean"}
RN50_yffcc15m = False  # @param{type:"boolean"}
RN50_cc12m = False  # @param{type:"boolean"}
RN50_quickgelu_yfcc15m = False  # @param{type:"boolean"}
RN50_quickgelu_cc12m = False  # @param{type:"boolean"}
RN101_yfcc15m = False  # @param{type:"boolean"}
RN101_quickgelu_yfcc15m = False  # @param{type:"boolean"}

# @markdown ####**Basic Settings:**

# NOTE steps可以改这里，需要重新初始化模型，我懒得改接口了orz
steps = 100  # @param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
tv_scale = 0  # @param{type: 'number'}
range_scale = 150  # @param{type: 'number'}
sat_scale = 0  # @param{type: 'number'}
cutn_batches = 1  # @param{type: 'number'}  # NOTE 这里会对图片做数据增强，累计计算n次CLIP的梯度，以此作为guidance。
skip_augs = False  # @param{type: 'boolean'}
# @markdown ####**Saving:**

intermediate_saves = 0  # @param{type: 'raw'}
intermediates_in_subfolder = True  # @param{type: 'boolean'}

# perlin_init = False  # @param{type: 'boolean'}
perlin_mode = 'mixed'  # @param ['mixed', 'color', 'gray']
set_seed = 'random_seed'  # @param{type: 'string'}
eta = 0.8  # @param{type: 'number'}
clamp_grad = True  # @param{type: 'boolean'}
clamp_max = 0.05  # @param{type: 'number'}

# EXTRA ADVANCED SETTINGS:
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05

# @markdown ---
cut_overview = "[12]*400+[4]*600"  # @param {type: 'string'}
cut_innercut = "[4]*400+[12]*600"  # @param {type: 'string'}
cut_ic_pow = "[1]*1000"  # @param {type: 'string'}
cut_icgray_p = "[0.2]*400+[0]*600"  # @param {type: 'string'}


# @markdown ####**Transformation Settings:**
use_vertical_symmetry = False  # @param {type:"boolean"}
use_horizontal_symmetry = False  # @param {type:"boolean"}
transformation_percent = [0.09]  # @param

display_rate = 3  # @param{type: 'number'}
n_batches = 1  # @param{type: 'number'}

# @markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False  # @param{type:"boolean"}
interp_spline = 'Linear'  # Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
resume_run = False
batch_size = 1


def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)


def wget(url, outputdir):
    res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def interp(t):
    return 3 * t**2 - 2 * t ** 3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=None):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size / max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size, args,
                 Overview=4,
                 InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2,
                 ):
        super().__init__()
        self.padargs = {}
        self.cutout_debug = False
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(input, ((sideY - max_size) // 2, (sideY - max_size) // 2, (sideX - max_size) // 2, (sideX - max_size) // 2), **self.padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if self.cutout_debug:
                # if is_colab:
                #     TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("/content/cutout_overview0.jpg",quality=99)
                # else:
                TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("cutout_overview0.jpg", quality=99)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if self.cutout_debug:
                # if is_colab:
                #     TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("/content/cutout_InnerCrop.jpg",quality=99)
                # else:
                TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("cutout_InnerCrop.jpg", quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def symmetry_transformation_fn(x):
    # NOTE 强制图像对称
    use_horizontal_symmetry = False
    if use_horizontal_symmetry:
        [n, c, h, w] = x.size()
        x = torch.concat((x[:, :, :, :w // 2], torch.flip(x[:, :, :, :w // 2], [-1])), -1)
        print("horizontal symmetry applied")
    if use_vertical_symmetry:
        [n, c, h, w] = x.size()
        x = torch.concat((x[:, :, :h // 2, :], torch.flip(x[:, :, :h // 2, :], [-2])), -2)
        print("vertical symmetry applied")
    return x


# def split_prompts(prompts):
#     prompt_series = pd.Series([np.nan for a in range(max_frames)])
#     for i, prompt in prompts.items():
#         prompt_series[i] = prompt
#     # prompt_series = prompt_series.astype(str)
#     prompt_series = prompt_series.ffill().bfill()
#     return prompt_series


"""
other chaos settings
"""
# dir settings

outDirPath = f'{PROJECT_DIR}/images_out'
createPath(outDirPath)
model_path = f'{PROJECT_DIR}/models'
createPath(model_path)


# GPU setup
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not useCPU) else 'cpu')
print('Using device:', DEVICE)
device = DEVICE  # At least one of the modules expects this name..
if not useCPU:
    if torch.cuda.get_device_capability(DEVICE) == (8, 0):  # A100 fix thanks to Emad
        print('Disabling CUDNN for A100 gpu', file=sys.stderr)
        torch.backends.cudnn.enabled = False

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
    'rescale_timesteps': True,
    'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
    'image_size': 512,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_checkpoint': use_checkpoint,
    'use_fp16': not useCPU,
    'use_scale_shift_norm': True,
})

model_default = model_config['image_size']
normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

# Make folder for batch
steps_per_checkpoint = steps + 10
# Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})


start_frame = 0
print('Starting Run:')
if set_seed == 'random_seed':
    random.seed()
    seed = random.randint(0, 2**32)
    # print(f'Using seed: {seed}')
else:
    seed = int(set_seed)

args = {
    # 'seed': seed,
    'display_rate': display_rate,
    'n_batches': n_batches,
    'batch_size': batch_size,
    'steps': steps,
    'diffusion_sampling_mode': diffusion_sampling_mode,
    # 'width_height': width_height,
    'tv_scale': tv_scale,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'cutn_batches': cutn_batches,
    # 'side_x': side_x,
    # 'side_y': side_y,
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
    'cut_overview': eval(cut_overview),
    'cut_innercut': eval(cut_innercut),
    'cut_ic_pow': eval(cut_ic_pow),
    'cut_icgray_p': eval(cut_icgray_p),
    'intermediate_saves': intermediate_saves,
    'intermediates_in_subfolder': intermediates_in_subfolder,
    'steps_per_checkpoint': steps_per_checkpoint,
    'set_seed': set_seed,
    'eta': eta,
    'clamp_grad': clamp_grad,
    'clamp_max': clamp_max,
    'skip_augs': skip_augs,
    'randomize_class': randomize_class,
    'clip_denoised': clip_denoised,
    'fuzzy_prompt': fuzzy_prompt,
    'rand_mag': rand_mag,
    'use_vertical_symmetry': use_vertical_symmetry,
    'use_horizontal_symmetry': use_horizontal_symmetry,
    'transformation_percent': transformation_percent,
}
args = SimpleNamespace(**args)

# ======================== GLOBAL SETTING END ========================


class Diffuser:
    def __init__(self, cutom_path='IDEA-CCNL/Taiyi-Diffusion-532M-Nature'):
        self.model_setup(cutom_path)

    def model_setup(self, custom_path):
        # LOADING MODEL
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        print(f'Prepping model...model name: {custom_path}')
        __, self.diffusion = create_model_and_diffusion(**model_config)
        self.model = HFUNetModel.from_pretrained(custom_path)
        # total = get_parameter_num(self.model)
        # print("Number of parameter: %.2fM" % (total/1e6))
        # print("Number of parameter: %.2fM" % (total/1024/1024))

        self.model.requires_grad_(False).eval().to(device)
        for name, param in self.model.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                param.requires_grad_()
        if model_config['use_fp16']:
            self.model.convert_to_fp16()
        print(f'Diffusion_model Loaded {diffusion_model}')

        # NOTE Directly Load The Text Encoder From Hugging Face
        print('Prepping model...model name: CLIP')
        self.taiyi_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
        self.taiyi_transformer = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval().to(device)
        self.clip_models = []
        if ViTB32:
            self.clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
        if ViTB16:
            self.clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device))
        if ViTL14:
            self.clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device))
        if ViTL14_336px:
            self.clip_models.append(clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device))
        print('CLIP Loaded')
        # self.lpips_model = lpips.LPIPS(net='vgg').to(device)

    def generate(self,
                 input_text_prompts=['夕阳西下'],
                 init_image=None,
                 skip_steps=10,
                 clip_guidance_scale=7500,
                 init_scale=2000,
                 st_dynamic_image=None,
                 seed=None,
                 side_x=512,
                 side_y=512,
                 ):

        seed = seed
        frame_num = 0
        init_image = init_image
        init_scale = init_scale
        skip_steps = skip_steps
        loss_values = []
        # if seed is not None:
        #     np.random.seed(seed)
        #     random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        #     torch.backends.cudnn.deterministic = True
        # target_embeds, weights = [], []
        frame_prompt = input_text_prompts

        print(f'Frame {frame_num} Prompt: {frame_prompt}')

        model_stats = []
        for clip_model in self.clip_models:
            # cutn = 16
            model_stat = {"clip_model": None, "target_embeds": [], "make_cutouts": None, "weights": []}
            model_stat["clip_model"] = clip_model

            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                # txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
                # NOTE use chinese CLIP
                txt = self.taiyi_transformer(self.taiyi_tokenizer(txt, return_tensors='pt')['input_ids'].to(device)).logits
                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0, 1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        init = None
        if init_image is not None:
            # init = Image.open(fetch(init_image)).convert('RGB')   # 传递的是加载好的图片。而非地址~
            init = init_image
            init = init.resize((side_x, side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]

                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                out = self.diffusion.p_mean_variance(self.model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)

                for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = int(t.item()) + 1  # errors on last step without +1, need to find source
                        # try:
                        input_resolution = model_stat["clip_model"].visual.input_resolution
                        # except:
                        #     input_resolution = 224

                        cuts = MakeCutoutsDango(input_resolution,
                                                Overview=args.cut_overview[1000 - t_int],
                                                InnerCrop=args.cut_innercut[1000 - t_int],
                                                IC_Size_Pow=args.cut_ic_pow[1000 - t_int],
                                                IC_Grey_P=args.cut_icgray_p[1000 - t_int],
                                                args=args,
                                                )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([args.cut_overview[1000 - t_int] + args.cut_innercut[1000 - t_int], n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item())  # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
                tv_losses = tv_loss(x_in)
                range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
                if init is not None and init_scale:
                    init_losses = self.lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if not torch.isnan(x_in_grad).any():
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and not x_is_NaN:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=args.clamp_max) / magnitude  # min=-0.02, min=-clamp_max,
            return grad

        if args.diffusion_sampling_mode == 'ddim':
            sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = self.diffusion.plms_sample_loop_progressive

        for i in range(args.n_batches):
            current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')

            batchBar = tqdm(range(args.n_batches), desc="Batches")
            batchBar.n = i
            batchBar.refresh()
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = self.diffusion.num_timesteps - skip_steps - 1
            # total_steps = cur_t

            if args.diffusion_sampling_mode == 'ddim':
                samples = sample_fn(
                    self.model,
                    (batch_size, 3, side_y, side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    eta=eta,
                    transformation_fn=symmetry_transformation_fn,
                    transformation_percent=args.transformation_percent
                )
            else:
                samples = sample_fn(
                    self.model,
                    (batch_size, 3, side_y, side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    order=2,
                )

            for j, sample in enumerate(samples):
                cur_t -= 1
                intermediateStep = False
                if args.steps_per_checkpoint is not None:
                    if j % steps_per_checkpoint == 0 and j > 0:
                        intermediateStep = True
                elif j in args.intermediate_saves:
                    intermediateStep = True
                if j % args.display_rate == 0 or cur_t == -1 or intermediateStep:
                    for k, image in enumerate(sample['pred_xstart']):
                        # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                        # percent = math.ceil(j / total_steps * 100)
                        if args.n_batches > 0:
                            filename = f'{current_time}-{parse_prompt(prompt)[0]}.png'
                        image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                        if j % args.display_rate == 0 or cur_t == -1:
                            image.save(f'{outDirPath}/{filename}')
                            if st_dynamic_image:
                                st_dynamic_image.image(image, use_column_width=True)
                            # self.current_image = image
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting")
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--text_scale', type=int, default=5000)
    parser.add_argument('--model_path', type=str, default="IDEA-CCNL/Taiyi-Diffusion-532M-Nature")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)

    user_args = parser.parse_args()

    dd = Diffuser(user_args.model_path)
    dd.generate([user_args.prompt],
                clip_guidance_scale=user_args.text_scale,
                side_x=user_args.width,
                side_y=user_args.height,
                )

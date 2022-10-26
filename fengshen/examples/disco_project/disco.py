from utils import *
import random
import json
import lpips
import gc
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import clip
from types import SimpleNamespace
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, create_gaussian_diffusion
from ipywidgets import Output
from datetime import datetime
from tqdm.notebook import tqdm
from glob import glob
import time
from transformers import PreTrainedModel
from guided_diffusion.unet import HFUNetModel, UNetConfig
import argparse

class Diffuser:
    def __init__(self, cutom_path='./nature_uncond_diffusion'):
        self.model_setup(cutom_path)

    def model_setup(self, custom_path):
        # LOADING MODEL
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        print(f'Prepping model...model name: {custom_path}')
        __, self.diffusion = create_model_and_diffusion(**model_config)
        self.model = HFUNetModel.from_pretrained(custom_path)

        self.model.requires_grad_(False).eval().to(device)
        for name, param in self.model.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                param.requires_grad_()
        if model_config['use_fp16']:
            self.model.convert_to_fp16()
        print(f'Diffusion_model Loaded {diffusion_model}')

        # NOTE Directly Load The Text Encoder From Hugging Face
        print(f'Prepping model...model name: CLIP')
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
        print(f'CLIP Loaded')
        # self.lpips_model = lpips.LPIPS(net='vgg').to(device)

    def generate(self, 
                    input_text_prompts=['夕阳西下'], 
                    init_image=None, 
                    skip_steps=10,
                    clip_guidance_scale=7500,
                    init_scale=2000,
                    st_dynamic_image=None,
                    seed = None,
                    side_x=512,
                    side_y=512,
                    ):

        seed = seed
        frame_num = 0
        init_image = init_image
        init_scale = init_scale
        skip_steps = skip_steps
        loss_values = []
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        target_embeds, weights = [], []
        frame_prompt = input_text_prompts


        print(f'Frame {frame_num} Prompt: {frame_prompt}')

        model_stats = []
        for clip_model in self.clip_models:
            cutn = 16
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
                        t_int = int(t.item())+1  # errors on last step without +1, need to find source
                        try:
                            input_resolution = model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution = 224

                        cuts = MakeCutoutsDango(input_resolution,
                                                Overview=args.cut_overview[1000-t_int],
                                                InnerCrop=args.cut_innercut[1000-t_int],
                                                IC_Size_Pow=args.cut_ic_pow[1000-t_int],
                                                IC_Grey_P=args.cut_icgray_p[1000-t_int],
                                                args=args,
                                                )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
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
                if torch.isnan(x_in_grad).any() == False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and x_is_NaN == False:
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
            total_steps = cur_t

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
                if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                    for k, image in enumerate(sample['pred_xstart']):
                        # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                        percent = math.ceil(j/total_steps*100)
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
    parser.add_argument('--model_path', type=str, default="wf-genius/nature_uncond_diffusion")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)

    user_args = parser.parse_args()


    dd = Diffuser(user_args.model_path)    
    dd.generate([user_args.prompt] , 
                clip_guidance_scale=user_args.text_scale,
                side_x=user_args.width,
                side_y=user_args.height,
                )
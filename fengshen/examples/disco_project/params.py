useCPU = False  # @param {type:"boolean"}
skip_augs = False  # @param{type: 'boolean'}
perlin_init = False  # @param{type: 'boolean'}

use_secondary_model = False  # @param {type: 'boolean'}    # set false if you want to use the custom model
diffusion_model = "custom"  # @param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]

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


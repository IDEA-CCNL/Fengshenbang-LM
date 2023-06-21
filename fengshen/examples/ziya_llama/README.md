# 全参数Finetune
这个示例主要用于全参数finetune Ziya-LLaMA-13B相关模型，目前支持数据并行+张量并行+ZeRO
## step0 环境安装
```
git clone git@github.com:IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM/
pip install --edit .
```

## step1 下载示例数据
[Ziya-Finetune-Small](https://huggingface.co/datasets/IDEA-CCNL/Ziya-Finetune-Small)，后续按照格式替换成自己的数据,目前代码直接用文件读取，非datasets读取，所以建议git clone下来然后在配置里引用对应的数据路径
```
git lfs install
git clone https://huggingface.co/datasets/IDEA-CCNL/Ziya-Finetune-Small
```

## step2 准备模型
以[Ziya-LLaMA-13B-Pretrain-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1)为例,因为开源的是delta参数的模型，首先按照指引合并模型,得到一个llama13b_hf的文件夹

## step3 将huggingface模型转化成fengshen的格式
需要自己指定convert_llama13b_to_fs.sh内地址
```
cd fengshen/examples/ziya_llama
sh convert_llama13b_to_fs.sh
```

## step4 请按照自己机器的显存大小进行tensor并行切分
这里提供两个跑起来的示例供参考
1. 使用 3*8=24张 3090(24GB),需要进行张量并行，Tensor parallesim = 8，这里需要手动进行模型转换
```
sh convert_llama13b_tp8.sh
```
2. 使用 1*8=8张 A100(80GB),不需要进行张量并行，Tensor parallesim = 1，这里不需要再进行模型转换

## step5 根据step4的两种配置分别进行训练
分别参考下面的脚本（这里采用slurm作为调度系统，如果没有，单机多卡训练去掉srun进行训练，多机多卡训练参考torchrun进行训练）

```
# 用8张80GB A100进行微调
sh finetune_no_tp.sh
```
```
# 用24张24GB 3090进行微调
sh finetune_tp.sh
```

训练loss曲线可以在封神榜公开的wandb项目查看
[ziya_llama13b_finetune_example](https://wandb.ai/fengshenbang/ziya_llama13b_finetune_example?workspace=user-1548988412)

## step6 验证微调后的生成效果
例如针对finetune_no_tp.sh微调出来的模型，验证生成效果，参考下面的脚本
```
sh generate_no_tp.sh
```

## TODO
1. 目前会出现过拟合现象，跟数据太少有关，另外也可以采用一些trick来缓解该问题
2. 增加一些下游评测和机器指标
3. 集成近期lora、qlora、lomo等高效训练的方法


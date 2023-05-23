# Taiyi-Stable-Diffusion Finetune示例

本示例可以应用于**IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1**在自建的数据集上进行进一步训练，同时稍微修改代码也能够兼容大部分Stable-Diffusion结构。本示例仅提供参考，有任何疑问或者有需要协助的都可以提Issue到本项目中，会有专门的同学解答~

注：已更新了[colab的example](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/finetune_taiyi_stable_diffusion/finetune_taiyi_stable_diffusion_example.ipynb)

## 数据处理

在./demo_dataset下有我们一个数据集的样例，其中一个sample由.jpg格式图片以及.txt文本文件组成，用户可以按照我们的格式处理然后直接将脚本内的datasets_path修改为自己的路径即可。(数据摘自[IDEA-CCNL/laion2B-multi-chinese-subset](https://huggingface.co/datasets/IDEA-CCNL/laion2B-multi-chinese-subset))

## 配置要求

Finetune **IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1** 十亿级别参数，我们自己测试所需要的配置基础如下。batch_size设定为1

fp32:

- 显存：26G以上
- 内存：64G以上

fp16:

- 显存：21G以上
- 内存：64G以上

fp16 + deepspeed offload

- 显存：6G以上
- 内存：80G以上

# 安装依赖
```shell
cd /home/wuxiaojun/workspace/before_llm/Fengshenbang-LM/fengshen/examples/finetune_taiyi_stable_diffusion
pip install -r requirements.txt
```

## 运行脚本

处理好自己的数据集后，只需要将脚本中的datasets_path指向你的数据集，不需要修改其他参数就能运行。在脚本中也提供了丰富的超参供大家修改，例如batch_size, ckpt_path等等都可以根据自己的需求做更改，其中model_path指向的是huggingface上的模型路径，下载可能比较慢，如果用户已经在本地下载过一份权重，直接将model_path改成本地路径即可。

一些常用的参数我们会放在[封神榜的文档里](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%B0%81%E7%A5%9E%E6%A1%86%E6%9E%B6/%E5%8F%82%E6%95%B0%E7%AE%A1%E7%90%86.html)

有任何不清楚的地方，不要吝啬你的Issue，直接提过来。

## 一些训练中的Trick

### Deepspeed

在示例中我们默认开始了Deepspeed，通过Deepspeed我们能提高不少训练效率（即使是单卡）。并且得益于Zero Redundancy Optimizer的技术，在多卡的环境我们能显著的减少显存占用，提高batch_size以获得更高的效率，强烈建议有条件的同学开启Deepspeed。

### 8BitAdam

TODO: 优化显存以及提高训练效率

[**中文**](./README.md) | [**English**](./README_en.md)

## 最新发布

* [更新T5预训练及下游任务示例](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/燃灯系列/Randeng-MegatronT5-770M.html)
* [更新BART预训练及下游任务示例](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/燃灯系列/BART-139M.html)

# 导航
  - [框架简介](#框架简介)
  - [依赖环境](#依赖环境)
  - [项目结构](#项目结构)
  - [设计思路](#设计思路)


## 框架简介

FengShen训练框架是封神榜大模型开源计划的重要一环，在大模型的生产和应用中起到至关重要的作用。FengShen可以应用在基于海量数据的预训练以及各种下游任务的finetune中。封神榜专注于NLP大模型开源，然而模型的增大带来不仅仅是训练的问题，在使用上也存在诸多不便。为了解决训练和使用的问题，FengShen参考了目前开源的优秀方案并且重新设计了Pipeline，用户可以根据自己的需求，从封神榜中选取丰富的预训练模型，同时利用FengShen快速微调下游任务。

通过我们的框架，你可以快速享受到：
1. 丰富的预训练、下游任务示例，一键开始训练
2. 配置化分布式逻辑，仅修改配置即可由单机到多机的加速
3. 单卡支持百亿级别内模型训练(Zero Optimizer优化)


## 依赖环境

* Python >= 3.6
* torch >= 1.1
* transformers >= 3.2.0
* pytorch-lightning >= 1.6.0

## 项目结构

```
├── data                        # 支持多种数据处理方式以及数据集
│   ├── cbart_dataloader
|   ├── fs_datasets             # 基于transformers datasets的封装，新增中文数据集(开源计划中)
|   ├── universal_datamodule    # 打通fs_datasets与lightning datamodule，减少重复开发工作量
│   ├── megatron_dataloader     # 支持基于Megatron实现的TB级别数据集处理、训练
│   ├── mmap_dataloader         # 通用的Memmap形式的数据加载
│   └── task_dataloader         # 支持多种下游任务
├── examples                    # 丰富的示例，从预训练到下游任务，应有尽有。
├── metric                      # 提供各种metric计算，支持用户自定义metric
├── losses                      # 同样支持loss自定义，满足定制化需求
├── tokenizer                   # 支持自定义tokenizer，比如我们使用的SentencePiece训练代码等
├── models                      # 模型库
│   ├── auto                    # 支持自动导入对应的模型
│   ├── bart
│   ├── longformer
│   ├── megatron_t5
│   └── roformer
└── utils                       # 实用函数
```

## 设计思路

FengShen框架目前整体基于Pytorch-Lightning & Transformer进行开发，在底层框架上不断开源基于中文的预训练模型，同时提供丰富的examples，每一个封神榜的模型都能找到对应的预训练、下游任务代码。

在FengShen上开发，整体可以按照下面的三个步骤进行：

1. 封装数据处理流程 -> pytorch_lightning.LightningDataModule
2. 封装模型结构 -> pytorch_lightning.LightningModule
3. 配置一些插件，比如log_monitor，checkpoint_callback等等。

一个完整的DEMO可以看Randeng-BART系列实例 -> [文档](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/燃灯系列/BART-139M.html) [代码](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/hf-ds/fengshen/examples/pretrain_bart)


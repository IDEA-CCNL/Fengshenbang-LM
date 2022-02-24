[**中文**](./README.md) | [**English**](./README_en.md)

# 导航
  - [框架简介](#框架简介)
  - [依赖环境](#依赖环境)
  - [项目结构](#项目结构)
  - [3分钟上手](#三分钟上手)


## 框架简介
FengShen训练框架是封神榜大模型开源计划的重要一环，在大模型的生产和应用中起到至关重要的作用。FengShen可以应用在基于海量数据的预训练以及各种下游任务的finetune中。封神榜专注于NLP大模型开源，然而模型的增大带来不仅仅是训练的问题，在使用上也存在诸多不便。为了解决训练和使用的问题，FengShen参考了目前开源的优秀方案并且重新设计了Pipeline，用户可以根据自己的需求，从封神榜中选取丰富的预训练模型，同时利用FengShen快速微调下游任务。

## 依赖环境

* Python >= 3.6
* torch >= 1.1
* transformers >= 3.2.0

## 项目结构

```
├── data                        # 支持多种数据处理方式以及数据集
│   ├── cbart_dataloader
│   ├── megatron_dataloader     # 支持基于Megatron实现的TB级别数据集处理、训练
│   ├── mmap_dataloader         # 通用的Memmap形式的数据加载
│   └── task_dataloader         # 支持多种下游任务
├── examples                    # 丰富的事例 快速上手
├── metric                      # 提供各种metric计算，支持用户自定义metric
├── losses                      # 同样支持loss自定义，满足定制化需求
├── tokenizer                   # 支持自定义tokenizer
├── models                      # 模型库
│   ├── auto                    # 支持自动导入对应的模型
│   ├── bart
│   ├── longformer
│   ├── megatron_t5
│   ├── roformer
│   ├── model_utils.py
│   └── transformer_utils.py
├── scripts                     # 第三方脚本
└── utils                       # 实用函数
```
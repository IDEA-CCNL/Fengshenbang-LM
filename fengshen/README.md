[**中文**](./README.md) | [**English**](./README_en.md)

# 导航
  - [框架简介](#框架简介)
  - [依赖环境](#依赖环境)
  - [项目结构](#项目结构)
  - [三分钟上手](#三分钟上手)


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

## 三分钟上手

这里展示如何使用FengShen框架对模型进行Finetune，这里demo使用封神榜开源的35亿参数模型[闻仲](https://github.com/IDEA-CCNL/Fengshenbang-LM)做一个知识问答的下游任务Finetune。
由于闻仲参数量过大，这里我们考虑使用Deepspeed对训练进行优化，使得能在单卡上对闻仲进行训练，这里测试使用一张A100进行。
Deepspeed相关文档可以参考 https://deepspeed.readthedocs.io/en/latest/

脚本位于scripts/fintune_wenzhong.sh，脚本内有部分slurm集群参数，用户可以根据需要保留或者删除。
几乎所有的训练参数都被涵盖在脚本内，用户仅需要调整部分参数（数据集路径等等）即可快速复现。通过两步能快速修改我们的finetune脚本到用户自定义的下游任务上。
整个训练的代码在fintune_wenzhong.py内。

### Step 1 实现自己DataModule & Module

DataModule主要是封装了各种数据处理的操作下里面，具体的文档可以参照lightning的[官方文档](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.datamodule.html?highlight=datamodule)
这里我们实现了一个QA的DataModule可供参考 data/task_dataloader/medicalQADataset.py
同时，用户需要对huggingface的model进行封装，在这里用户可以自定义metrics的计算方式、validation等等，采用LightningModule的方式进行封装，同样可以参考lightning的[官方文档](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=LightningModule)
我们这里依旧提供了一个demo，可以参照fintune_wenzhong.py，针对下游不同的任务进行不同的封装。

### Step 2 修改scrpit参数

在scripts/fintune_wenzhong.sh内涵盖了此次训练的所有参数，包括learning rate、deepspeed配置等等。利用Deepspeed stage 3我们可以轻松在单卡上进行闻仲的下游finetune。

目前项目仍在快速推进当中，更多demo敬请期待


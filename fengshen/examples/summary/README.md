# randeng pegasus 摘要模型 Finetune

***
本模型主要实现了pegasus结构的燃灯系列摘要模型，通过使用LCSTS、CSL中文数据集文本-摘要对进行finetune，pegasus模型是专门针对摘要抽取任务训练的预训练模型。在下游任务的数据上，仅通过少量的数据进行微调就能达到良好的摘要抽取效果。

### finetune细节

#### 模型：

finetune的模型是燃灯模型，燃灯模型是pegasus结构，在预训练阶段主要是使用wudao数据进行的预训练，主要以中文语料为主。模型参数量总共为5亿，主要参数如下所示：

| 配置 | 参数 |
| ---- | ---- |
| encoder layers | 16 |
| encoder_attention_heads | 16 |
| encoder_ffn_dim | 4096 |
| decoder layers | 16 |
| decoder_attention_heads| 16 |
| decoder_ffn_dim | 4096 |
| max_encode_length | 128 |
| max_decode_length | 64 |

#### 数据

用于finetune的LCSTS文本-标题对数据，格式如下：
```
.....
{'text': '..........', 'summary': '...........'}
{'text': '..........', 'summary': '...........'}
.....
```

#### 框架

finetune 使用的是IDEA研究院认知计算小组的开源框架-封神，具体fintune代码可看 finetune_pegasus.py

#### finetune参数

finetune阶段使用了deepspeed来加速训练
| dataset | Learning rate | Batch size | Beam size |  Max input tokens | Max target tokens |
| ---- | ---- | ---- | ---- | ---- | ---- |
| LCSTS | 5e-5 | 128 | 8 | 128 | 64 |

其他训练参数请看randeng_pegasus_523M_summary.sh


### finetune后模型效果

1. LCSTS摘要数据finetune后效果

| model | rouge-1 | rouge-2 | rouge-L |
| ---- | ---- | ---- | ---- |
| Pegasus-base  | 44.13 | 31.31 | 41.06 | 
| Pegasus-large | 49.42 | 37.91 | 46.63 |


### 使用方式
可直接通过Hugging face或者pytoch-ligthning框架调用。下面给出的例子是hugging face的调用方法：



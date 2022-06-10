# 燃灯系列-Pegasus摘要模型预训练
Pegasus预训练模型是专门为摘要任务而设计的预训练模型，相比于其它通用预训练模型，Pegasus 模型的架构设计更贴近下游的摘要任务，在摘要抽取的效果上的表现相比其他通用模型表现更好

### 模型架构和参数
Pegasus的模型架构是标准的encoder-decoder的Transformer结构，训练任务是用的是GSG（ Gap Sentences Generation）任务。GSG任务主要是通过对文本中的重要的句子进行mask，然后再通过decoder恢复。模型详细参数可看config.json

1. base版本

| 配置 | 参数 |
| ---- | ---- |
| encoder layers | 12 |
| encoder_attention_heads | 12 |
| encoder_ffn_dim | 3072 |
| decoder layers | 12 |
| decoder_attention_heads| 12 |
| decoder_ffn_dim | 3072 |
| max_encode_length | 512 |

2. large 版本
   
| 配置 | 参数 |
| ---- | ---- |
| encoder layers | 16 |
| encoder_attention_heads | 16 |
| encoder_ffn_dim | 4096 |
| decoder layers | 16 |
| decoder_attention_heads| 16 |
| decoder_ffn_dim | 4096 |
| max_encode_length | 1024 |

### 训练数据
训练数据使用的是wudao 180g数据。数据进行了简单的预处理包括：
1. 过滤过长单句（这样的句子通常会包括一些乱码句，无上下文语义的列表句、各种符号句，歌词句等）
2. 过滤句子数过少文本，如句子数少于3句则抛弃

### 模型

pegasus-base: [Randeng_pegasus_238M_summary](https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_238M_Summary) <br/>
pegasus-large: [Randeng_pegasus_523M_summary](https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_523M_Summary)

主要文件：
- tokenizers_pegasus.py 中文版pegasus的tokenize实现
- pretrain_pegasus.py 模型预训练的核心实现文件
- pretrain_pegasusu.sh 预训练脚本，具体参数可通过此脚本修改
- data_utils.py 模型的一些工具代码

### 下游效果

#### LCSTS摘要数据finetune后效果

| model | rouge-1 | rouge-2 | rouge-L |
| ---- | ---- | ---- | ---- |
| Pegasus-base  | 44.13 | 31.31 | 41.06 | 
| Pegasus-large | 49.42 | 37.91 | 46.63 |
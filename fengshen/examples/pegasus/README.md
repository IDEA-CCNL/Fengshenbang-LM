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

#### 使用方式
可直接通过Hugging face或者pytoch-ligthning框架调用。下面给出的例子是hugging face的调用方法：
```python
from transformers import PegasusForConditionalGeneration
# Need to download tokenizers_pegasus.py and other Python script from Fengshenbang-LM github repo in advance,
# or you can mv download in tokenizers_pegasus.py and data_utils.py in https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_238M_Summary/tree/main
# Stronly recomend you git clone the Fengshenbang-LM repo:
# 1. git clone https://github.com/IDEA-CCNL/Fengshenbang-LM
# 2. cd Fengshenbang-LM/fengshen/examples/pegasus/
# and then you will see the tokenizers_pegasus.py and data_utils.py which are needed by pegasus model
from tokenizers_pegasus import PegasusTokenizer

model = PegasusForConditionalGeneration.from_pretrained("IDEA-CCNL/randeng_pegasus_238M_summary")
tokenizer = PegasusTokenizer.from_pretrained("path/to/vocab.txt")

text = "在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！"
inputs = tokenizer(text, max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

### 下游效果

#### LCSTS摘要数据finetune后效果

| model | rouge-1 | rouge-2 | rouge-L |
| ---- | ---- | ---- | ---- |
| Pegasus-base  | 44.13 | 31.31 | 41.06 | 
| Pegasus-large | 49.42 | 37.91 | 46.63 |


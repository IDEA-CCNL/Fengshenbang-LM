# <center> yuyuanQA模型finetune
本示例主要实现了基于GPT2结构的Yuyuan医疗大模型，通过医疗问答对Finetune，使大模型能够有closebook-qa的能力。
### 数据和模型
#### 模型：
finetune的模型是yuyuan模型，余元模型是GPT2的结构，在预训练阶段主要是用PubMed医疗相关的数据集进行的预训练。是一个医疗领域的大模型。模型共有35亿参数，主要参数如下表所示：

|    配置     | 参数  |
| :---------: | :---: |
|   nlayers   |  30   |
|  nheaders   |  32   |
| hidden-size | 3072  |
| seq-length  | 1024  |

预训练的数据，主要医疗相关的论文、杂志期刊等，以英文语料为主。
#### 数据：
用于finetune的语料是清洗于[MedQuAD](https://github.com/abachaa/MedQuAD)数据集，清洗完成后是下面的格式：
```text
......
{'question':'.........','answer':'........'}
{'question':'.........','answer':'........'}
......
```
### finetune框架以及参数配置
#### 框架 ：
finetune的框架是IDEA研究院CCNL小组整合各大框架的优点开源的[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)，具体代码可以参考[inetune_medicalQA.py](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/dev_wzw/fengshen/examples/wenzhong_qa/finetune_medicalQA.py)和[medicalQADataset.py](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/dev_wzw/fengshen/data/task_dataloader/medicalQADataset.py)。
#### 训练参数：
训练参数，我们采用了deepspeed相关的配置，用2个集群的节点共16张A100，在很短的时间内完成了finetune。具体参数配置可以参考[finetune_GPT2_medicalQA.sh](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/dev_wzw/fengshen/examples/wenzhong_qa/finetune_GPT2_medicalQA.sh)
### finetune后的效果以及使用
#### 效果对比：
finetune后的模型，用100对问答对，基于BLEU分与之前用Magetron框架训练的模型进行了简单的对比，效果比较接近。

unsmoth method:
| 框架     | 1-gram             | 2-gram             | 3-gram             | 4-gram              |
| -------- | ------------------ | ------------------ | ------------------ | ------------------- |
| Fengshen | 0.5241376169070796 | 0.5215762466122144 | 0.4894353584800885 | 0.44840139357073466 |
| Magetron | 0.5321340489166898 | 0.5110257474778213 | 0.4703745962926368 | 0.4310875933354554  |

smoth method:
| 框架     | 1-gram            | 2-gram             | 3-gram             | 4-gram             |
| -------- | ----------------- | ------------------ | ------------------ | ------------------ |
| Fengshen | 0.717829796617609 | 0.6516910802858905 | 0.5859726677095979 | 0.525510691686505  |
| Magetron | 0.776190980974117 | 0.6749801211321476 | 0.5897846253142169 | 0.5230773076722481 |
#### 使用方式：
支持直接用Haggingface或者pytorch-lightning框架调用。由于在finetune的时候，加入了prompt，在问答的时候，输入应该是：
```python 
Question:your question about medical? answer:
```
接着模型就回以续写的方式回答你的问题。
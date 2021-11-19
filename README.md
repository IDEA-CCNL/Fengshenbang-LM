# Fengshenbang-LM
封神榜-LM是IDEA认知计算中心主导的基础大模型开源计划,我们计划从模型结构、模型尺寸、专业领域三个维度去开发基础大模型，并逐步开源我们最新的研究成果。
预训练大模型是认知智能和自然语言的基础设施。不同的模型结构，不同的模型尺寸，再加上不同的专业领域，预训练大模型构成了一个巨大的空间。大模型训练需要昂贵的算力和高技术人才，训练一个适用于各自领域任务的大模型对小团队和小公司来说是巨大的挑战。在这个空间中，目前只填充了为数不多的模型，这些模型中，又只有少数是开源的。

为了全方面建设中文自然语言的基础设施，沈向洋博士在IDEA研究院宣布，我们开启一个“封神榜”大模型开源计划。
*我们将覆盖不同的模型结构、不同的模型尺寸、不同的专业领域，全谱系的开放多个大模型系列。
*我们也会持续的对这些模型进行升级，持续在模型规模、知识融入、监督任务辅助等方向不断优化，保持最新的训练数据和最新的训练算法，让这些大模型始终居于领先地位。
*我们也希望各个公司、高校、机构跟我们合作，一起共建大模型开源体系。
*希望我们的一起努力，可以推动中文认知智能和自然语言的深入发展和产业落地。

![avatar](models.png)
  
## 二郎神系列

Encoder结构为主的双向语言模型，专注于解决各种自然语言理解任务。
13亿参数的二郎神-1.3B大模型，采用280G数据，32张A100训练14天，是最大的开源中文Bert大模型。2021年11月10日在中文语言理解权威评测基准FewCLUE 榜单上登顶。其中，CHID(成语填空)、TNEWS(新闻分类)超过人类，CHID(成语填空)、CSLDCP(学科文献分类)、OCNLI(自然语言推理)单任务第一，刷新小样本学习记录。二郎神系列会持续在模型规模、知识融入、监督任务辅助等方向不断优化。
详细介绍见 https://mp.weixin.qq.com/s/bA_9n_TlBE9P-UzCn7mKoA
![image](https://user-images.githubusercontent.com/4384420/141752311-d15c2a7f-cd83-4e9e-99a5-cb931088845e.png)



### 模型下载地址
[二郎神-1.3B](https://big-models.obs.cn-north-4.myhuaweicloud.com:443/%E4%BA%8C%E9%83%8E%E7%A5%9E-1.3B.zip?AccessKeyId=UFREDVP4MG5MSSDPRU0V&Expires=1668225215&Signature=aCDiVHK6xIiLnrLTWLa2ysKRcRY%3D)

### 使用示例
提供下载
``` python
from transformers import MegatronBertConfig, MegatronBertModel
from transformers import BertTokenizer
import torch

model_pretrained_weight_path='/home/'  #模型的权重路径
config=MegatronBertConfig.from_pretrained(model_pretrained_weight_path)
model=MegatronBertModel.from_pretrained(model_pretrained_weight_path)
text = "北京是中国的首都"
encoded_input = torch.tensor([tokenizer.encode(text)])
output = model(encoded_input)
```

### 下游效果
|     模型   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| 二郎神-1.3B | 0.7608      |   0.5996    | 0.6234      |   0.7917    | 0.81    | 0.9243    | 0.872    |

## 周文王系列
IDEA研究院认知计算中心联合追一科技有限公司的新结构大模型。该模型在训练阶段就统一考虑LM（Language Model）和MLM（Mask Language Model）任务，增加了旋转位置编码技术，让模型同时具备生成和理解的能力。目前已有13亿参数的周文王-1.3B大模型，是中文领域同时做LM和MLM任务最大的模型，会持续在模型规模、知识融入、监督任务辅助等方向不断优化。

周文王-1.3B: 13亿参数 

### 模型下载地址
[周文王-1.3B](https://big-models.obs.cn-north-4.myhuaweicloud.com:443/%E5%91%A8%E6%96%87%E7%8E%8B-1.3B.zip?AccessKeyId=UFREDVP4MG5MSSDPRU0V&Expires=1668225200&Signature=5azS%2BtqThr0MiFtWULwM2tE/Tug%3D)

### 使用示例
模型加载方法

``` python
from roformer.modeling_roformer import RoFormerModel            #从本仓库提供的roformer文件中导入roformer模型
from roformer.configuration_roformer import RoFormerConfig
from transformers import BertTokenizer
import torch

model_pretrained_weight_path='./roformer_v1/'  #预训练模型权重路径
tokenizer = BertTokenizer.from_pretrained(model_pretrained_weight_path)
model = RoFormerModel.from_pretrained(model_pretrained_weight_path)
text = "北京是中国的首都"
encoded_input = torch.tensor([tokenizer.encode(text)])
output = model(encoded_input)
print(output)
```

### 下游效果

#### NLU

|     模型   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext | 0.7406      |   0.575    | 0.6035      |   0.743    | 0.7973    | 0.8348    | 0.857    |
| 周文王-110M | 0.7258      |   0.5698    | 0.5905      |   0.728    | 0.7569    | 0.6438    | 0.8283    |
| 周文王-1.3B | 0.      |   0.    | 0.     |   0.    | 0.    | 0.    | 0.    |

#### NLG

## 闻仲系列
Decoder结构为主的单向语言模型，是一系列强大的生成模型。
35亿参数的闻仲-3.5B大模型，采用100G数据，256张A100训练28小时。

### 使用示例
``` python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model_pretrained_weight_path='/home/'  #模型的权重路径
tokenizer = GPT2Tokenizer.from_pretrained(model_pretrained_weight_path)
model = GPT2LMHeadModel.from_pretrained(model_pretrained_weight_path)
device = torch.device("cuda:6")
model.to(device)
model.eval()
model.half()
text = "北京是中国的首都"
encoded_input = torch.tensor([tokenizer.encode(text)]).to(device)
output = model.generate(input_ids=encoded_input, max_length=32, num_return_sequences=1)
for sentence in output:
    sentence = sentence.tolist()
    text = tokenizer.decode(sentence, clean_up_tokenization_spaces=True)
    print(text)

```

### 模型下载地址
[闻仲-3.5B](https://big-models.obs.cn-north-4.myhuaweicloud.com:443/%E9%97%BB%E4%BB%B2-3.5B.zip?AccessKeyId=UFREDVP4MG5MSSDPRU0V&Expires=1668225244&Signature=1HSpdcstUw2DMuiOufYRRN8fPio%3D)

## 燃灯
Transformer结构为主的编解码语言模型，7.7亿参数的燃灯-7.7B大模型，采用280G数据，16张A100训练14天。
敬请期待

## 余元
医学领域的余元系列，35亿参数余元-3.5B大模型，采用50G的医疗领域数据和知识，在已有的通用模型基础上继续训练，256张A100训练28小时，应用于医学领域的生成和判定任务。


### 模型下载地址
[余元-3.5B](https://big-models.obs.cn-north-4.myhuaweicloud.com:443/%E4%BD%99%E5%85%83-3.5B.zip?AccessKeyId=UFREDVP4MG5MSSDPRU0V&Expires=1668225231&Signature=C8Ai5VAyd4/ZPPWnIjEXi0wP1U8%3D)

## FAQ

## License

[Apache License 2.0](LICENSE)

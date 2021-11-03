# Fengshenbang-LM
封神榜-LM是IDEA认知计算中心主导的基础大模型开源计划,我们计划从模型结构、模型规模、应用领域三个维度去开发基础大模型，并逐步开源我们最新的研究成果。

目前已经开源了二郎神系列模型和雷震子系列模型，后续还会陆续开源姜子牙、周文王、周武王、闻仲、金吒、木吒、哪吒、申公豹、比干等系列。 
  
## 二郎神系列
BERT结构，训练过程融入知识，下游任务

二郎神-xlarge: 13亿参数

### 模型下载地址
[二郎神-1.3B]

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
| roberta-chinese-roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| 杨戬-xlarge | 0.7608      |   0.5996    | 0.6234      |   0.7917    | 0.81    | 0.9243    | 0.872    |

## 雷震子系列
BERT结构，训练过程同时考虑LM和MLM任务，采用roformer位置编码

雷震子-base： 1亿参数

雷震子-1.3B: 13亿参数 

### 模型下载地址
[雷震子-1.3B]

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
| roberta-chinese-roberta-wwm-ext | 0.7406      |   0.575    | 0.6035      |   0.743    | 0.7973    | 0.8348    | 0.857    |
| 雷震子-base | 0.7258      |   0.5698    | 0.5905      |   0.728    | 0.7569    | 0.6438    | 0.8283    |
| 雷震子-xlarge | 0.      |   0.    | 0.     |   0.    | 0.    | 0.    | 0.    |

#### NLG



## 姜子牙系列
T5结构

## 周文王
医疗领域模型，GPT结构

周文王-3.5B: 35亿参数

## 周武王
金融领域模型，BERT结构

## 闻仲
法律领域模型，T5结构



## FAQ


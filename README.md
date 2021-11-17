# Fengshenbang-LM
封神榜-LM是IDEA认知计算中心主导的基础大模型开源计划,我们计划从模型结构、模型规模、应用领域三个维度去开发基础大模型，并逐步开源我们最新的研究成果。

目前已经开源了二郎神系列模型和周文王系列模型，后续还会陆续开源闻仲、余元、燃灯等系列。 
![avatar](models.png)
  
## 二郎神系列
Encoder结构，训练过程融入知识，增加有监督任务二次预训练。

二郎神-1.3B: 13亿参数，目前中文开源中最大的BERT结构模型，FewCLUE总分登顶榜一，CHID(成语填空)、TNEWS(新闻分类)超过人类，CHID (成语填空) 、CSLDCP(学科文献分类)、OCNLI(自然语言推理)单任务第一
![image](https://user-images.githubusercontent.com/4384420/141752311-d15c2a7f-cd83-4e9e-99a5-cb931088845e.png)


### 模型下载地址
[二郎神-1.3B](https://big-models.obs.cn-north-4.myhuaweicloud.com:443/%E4%BA%8C%E9%83%8E%E7%A5%9E-1.3B.zip?AccessKeyId=UFREDVP4MG5MSSDPRU0V&Expires=1668225215&Signature=aCDiVHK6xIiLnrLTWLa2ysKRcRY%3D)

### 模型加载
``` python
from transformers import MegatronBertConfig, MegatronBertModel
from transformers import BertTokenizer
import torch

model_pretrained_weight_path='/home/'  #模型的权重路径
tokenizer = BertTokenizer.from_pretrained(model_pretrained_weight_path)
config=MegatronBertConfig.from_pretrained(model_pretrained_weight_path)
model=MegatronBertModel.from_pretrained(model_pretrained_weight_path)

```
### 使用示例
为了便于开发者快速使用我们的开源模型，这里提供了一个下游任务的finetune示例脚本，使用的[CLUE](https://github.com/CLUEbenchmark/CLUE)上的afqmc语义匹配任务数据，运行脚本如下。其中data_path为数据路径，afqmc任务数据的[下载地址](https://github.com/CLUEbenchmark/CLUE)，pretrained_model_path为预训练模型的路径。
``` sh
python example/finetune.py " \
        --train_data_path $TRAIN_DATA_PATH \
        --dev_data_path $DEV_DATA_PATH \
        --test_data_path $TSET_DATA_PATH \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --checkpoints ./model.pth \
        --output_path ./afqmc_predict.json \
        --log_file_path ./finetune.log \
        --batch_size 32 \
        --learning_rate 0.00002 \
        --max_length 64 \
        --epoch 7 \
        --model_type megatron \
            "
```
为了便于开发者在开源模型的基础上继续做任务相关的预训练，这里提供了一个继续预训练的pretraining脚本，运行脚本如下：
``` sh
python example/pretraining.py " \
        --train_data_path $TRAIN_DATA_PATH \
        --dev_data_path $DEV_DATA_PATH \
        --test_data_path $TSET_DATA_PATH \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --checkpoints ./model.pth \
        --output_path ./afqmc_predict.json \
        --log_file_path ./pretraining.log \
        --batch_size 128 \
        --learning_rate 0.00002 \
        --max_length 64 \
        --epoch 135 \
        --model_type megatron \
            "
```



### 下游效果
|     模型   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| 二郎神-1.3B | 0.7608      |   0.5996    | 0.6234      |   0.7917    | 0.81    | 0.9243    | 0.872    |

## 周文王系列
周文王采用的是Unified结构，即训练过程同时考虑LM和MLM任务，且采用roformer位置编码。因此周文王系列模型既可以做自然语言理解任务，也可以做自然语言生成任务。


### 模型下载地址
[周文王-1.3B](https://big-models.obs.cn-north-4.myhuaweicloud.com:443/%E5%91%A8%E6%96%87%E7%8E%8B-1.3B.zip?AccessKeyId=UFREDVP4MG5MSSDPRU0V&Expires=1668225200&Signature=5azS%2BtqThr0MiFtWULwM2tE/Tug%3D)

### 模型加载
由于HuggingFace没有现成的双任务RoFormer模型结构。因此需要从本仓库model文件夹中提供的脚本导入。导入示例如下：
``` python
from model.roformer.modeling_roformer import RoFormerModel            #从本仓库提供的roformer文件中导入roformer模型
from model.roformer.configuration_roformer import RoFormerConfig
from transformers import BertTokenizer
import torch

model_pretrained_weight_path='./home/'  #预训练模型权重路径
tokenizer = BertTokenizer.from_pretrained(model_pretrained_weight_path)
config = model = RoFormerConfig.from_pretrained(model_pretrained_weight_path)
model = RoFormerModel.from_pretrained(model_pretrained_weight_path)
```


### 使用示例

``` sh
python example/finetune.py " \
        --train_data_path $TRAIN_DATA_PATH \
        --dev_data_path $DEV_DATA_PATH \
        --test_data_path $TSET_DATA_PATH \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --checkpoints ./model.pth \
        --output_path ./afqmc_predict.json \
        --log_file_path ./finetune.log \
        --batch_size 32 \
        --learning_rate 0.00002 \
        --max_length 64 \
        --epoch 7 \
        --model_type roformer \
            "
```

### 下游效果

#### 自然语言理解
使用周文王-1.3B模型进行自然语言理解任务时，需要将token_type全部设置为0

|     模型   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| 周文王-1.3B | 0.7463     |   0.6036    | 0.6288     |   0.7654   | 0.7741    | 0.8849    | 0. 8777   |

#### 自然语言生成
使用周文王-1.3B模型进行自然语言生成任务时，需要将token_type全部设置为1




## 闻仲系列
Decoder结构

### 模型下载
[闻仲-3.5B](https://big-models.obs.cn-north-4.myhuaweicloud.com:443/%E9%97%BB%E4%BB%B2-3.5B.zip?AccessKeyId=UFREDVP4MG5MSSDPRU0V&Expires=1668225244&Signature=1HSpdcstUw2DMuiOufYRRN8fPio%3D)


## 燃灯
transformer结构

## 余元
医疗领域模型

### 模型下载
[余元-3.5B](https://big-models.obs.cn-north-4.myhuaweicloud.com:443/%E4%BD%99%E5%85%83-3.5B.zip?AccessKeyId=UFREDVP4MG5MSSDPRU0V&Expires=1668225231&Signature=C8Ai5VAyd4/ZPPWnIjEXi0wP1U8%3D)

## FAQ


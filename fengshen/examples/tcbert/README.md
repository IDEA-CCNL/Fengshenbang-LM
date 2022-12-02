[**中文**](./README.md)

# TCBert
论文 《[TCBERT: A Technical Report for Chinese Topic Classification BERT](https://arxiv.org/abs/2211.11304)》源码

## Requirements

安装 fengshen 框架

```shell
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable .
```

## Quick Start

你可以参考我们的 [example.py](./example.py) 脚本，只需要将处理好的 ```train_data```、```dev_data```、```test_data```、 ```prompt```、```prompt_label``` ，输入模型即可。
```python
import argparse
from fengshen.pipelines.tcbert import TCBertPipelines
from pytorch_lightning import seed_everything

total_parser = argparse.ArgumentParser("Topic Classification")
total_parser = TCBertPipelines.piplines_args(total_parser)
args = total_parser.parse_args()
    
pretrained_model_path = 'IDEA-CCNL/Erlangshen-TCBert-110M-Classification-Chinese'
args.learning_rate = 2e-5
args.max_length = 512
args.max_epochs = 3
args.batchsize = 1
args.train = 'train'
args.default_root_dir = './'
# args.gpus = 1   #注意：目前使用CPU进行训练，取消注释会使用GPU，但需要配置相应GPU环境版本
args.fixed_lablen = 2 #注意：可以设置固定标签长度，由于样本对应的标签长度可能不一致，建议选择合适的数值表示标签长度

train_data = [
        {"content": "凌云研发的国产两轮电动车怎么样，有什么惊喜？", "label": "科技",}
    ]

dev_data = [
    {"content": "我四千一个月，老婆一千五一个月，存款八万且有两小孩，是先买房还是先买车？","label": "汽车",}
]
    
test_data = [
    {"content": "街头偶遇2018款长安CS35，颜值美炸！或售6万起，还买宝骏510？"}
]

prompt = "下面是一则关于{}的新闻："

prompt_label = {"汽车":"汽车", "科技":"科技"}

model = TCBertPipelines(args, model_path=pretrained_model_path, nlabels=len(prompt_label))

if args.train:
    model.train(train_data, dev_data, prompt, prompt_label)
result = model.predict(test_data, prompt, prompt_label)
```


## Pretrained Model
为了提高模型在话题分类上的效果，我们收集了大量话题分类数据进行基于`prompt`的预训练。我们已经将预训练模型开源到 ```HuggingFace``` 社区当中。

| 模型 | 地址   |
|:---------:|:--------------:|
| Erlangshen-TCBert-110M-Classification-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-110M-Classification-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-110M-Classification-Chinese)   |
| Erlangshen-TCBert-330M-Classification-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-330M-Classification-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-330M-Classification-Chinese)       |
| Erlangshen-TCBert-1.3B-Classification-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-1.3B-Classification-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-1.3B-Classification-Chinese)   |
| Erlangshen-TCBert-110M-Sentence-Embedding-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-110M-Sentence-Embedding-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-110M-Sentence-Embedding-Chinese)       |
| Erlangshen-TCBert-330M-Sentence-Embedding-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese)       |
| Erlangshen-TCBert-1.3B-Sentence-Embedding-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-1.3B-Sentence-Embedding-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-1.3B-Sentence-Embedding-Chinese)       |

## Experiments

对每个不同的数据集，选择合适的模板```Prompt```
Dataset      | Prompt    
|------------|------------|
| TNEWS | 下面是一则关于{}的新闻：       |
| CSLDCP | 这一句描述{}的内容如下：       |
| IFLYTEK | 这一句描述{}的内容如下：       |

使用上述```Prompt```的实验结果如下：
| Model      | TNEWS    | CLSDCP   | IFLYTEK     |  
|------------|------------|----------|-----------|
| Macbert-base | 55.02       | 57.37     | 51.34        | 
| Macbert-large | 55.77	     | 58.99     | 	50.31         | 
| Erlangshen-1.3B | 57.36       | 62.35     | 53.23       | 
| TCBert-base-110M-Classification-Chinese | 55.57       | 58.60     | 49.63        | 
| TCBert-large-330M-Classification-Chinese | 56.17       | 61.23     | 51.34        | 
| TCBert-1.3B-Classification-Chinese | 57.41       | 65.10    | 53.75        | 
| TCBert-base-110M-Sentence-Embedding-Chinese | 54.68       | 59.78     | 49.40        | 
| TCBert-large-330M-Sentence-Embedding-Chinese | 55.32       | 62.07     | 51.11        | 
| TCBert-1.3B-Sentence-Embedding-Chinese | 57.46       | 65.04     | 53.06        | 

## Dataset

需要您提供：```训练集```、```验证集```、```测试集```、```Prompt```、```标签映射```五个数据，对应的数据格式如下：

#### 训练数据 示例
必须包含```content```和```label```字段
```json
[{
    "content": "街头偶遇2018款长安CS35，颜值美炸！或售6万起，还买宝骏510？",   
    "label": "汽车"
}]
```

#### 验证数据 示例
必须包含```content```和```label```字段
```json
[{
    "content": "宁夏邀深圳市民共赴“寻找穿越”之旅",
    "label": "旅游"
}]
```

#### 测试数据 示例
必须包含```content```字段
```json
[{
    "content": "买涡轮增压还是自然吸气车？今天终于有答案了！"
}]
```
#### Prompt 示例
可以选择任一模版，模版的选择会对模型效果产生影响，其中必须包含```{}```，作为标签占位符
```json
"下面是一则关于{}的新闻："
```

#### 标签映射 示例
可以将真实标签映射为更合适Prompt的标签，支持映射后的标签长度不一致
```json
{
    "汽车": "汽车", 
    "旅游": "旅游", 
    "经济生活": "经济生活",
    "房产新闻": "房产"
}
```

## License

[Apache License 2.0](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/LICENSE)


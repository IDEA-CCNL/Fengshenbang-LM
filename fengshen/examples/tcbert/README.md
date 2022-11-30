[**中文**](./README.md)

## Requirements

安装 fengshen 框架

```shell
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable .
```

## Quick Start

你可以参考我们的 [example.py](./example.py) 脚本，只需要将处理好的 train、dev、test 即输入模型即可。
```python
import argparse
from fengshen.pipelines.tcbert import TCBertPipelines
from pytorch_lightning import seed_everything

total_parser = argparse.ArgumentParser("TASK NAME")
total_parser = TCBertPipelines.piplines_args(total_parser)
args = total_parser.parse_args()
    
pretrained_model_path = 'IDEA-CCNL/Erlangshen-UniMC-RoBERTa-110M-Chinese'
args.learning_rate = 2e-5
args.max_length = 512
args.max_epochs = 3
args.batchsize = 4
args.train = 'train'
args.default_root_dir = './'

train_data = [
        {"content": "凌云研发的国产两轮电动车怎么样，有什么惊喜？", "label": "科技",}
    ]

dev_data = [
    {"content": "我四千一个月，老婆一千五一个月，存款八万且有两小孩，是先买房还是先买车？","label": "汽车",}
]
    
test_data = [
    {"content": "街头偶遇2018款长安CS35，颜值美炸！或售6万起，还买宝骏510？"}
]

prompt_label = {"汽车":"汽车",  "科技":"科技"}

model = TCBertPipelines(args, model_path=pretrained_model_path, nlabels=len(prompt_label))

if args.train:
	model.train(train_data, dev_data)
result = model.predict(test_data)
```
## Pretrained Model
我们使用收集的主题类数据集在二郎神模型的基础上进行进一步预训练。我们已经将预训练模型开源到 HuggingFace 社区当中。

| 模型 | 地址   |
|:---------:|:--------------:|
| Erlangshen-TCBert-110M-Classification-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-110M-Classification-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-110M-Classification-Chinese)   |
| Erlangshen-TCBert-330M-Classification-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-330M-Classification-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-330M-Classification-Chinese)       |
| Erlangshen-TCBert-1.3B-Classification-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-1.3B-Classification-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-1.3B-Classification-Chinese)   |
| Erlangshen-TCBert-110M-Sentence-Embedding-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-110M-Sentence-Embedding-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-110M-Sentence-Embedding-Chinese)       |
| Erlangshen-TCBert-330M-Sentence-Embedding-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese)       |
| Erlangshen-TCBert-1.3B-Sentence-Embedding-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-1.3B-Sentence-Embedding-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-TCBert-1.3B-Sentence-Embedding-Chinese)       |


## Dataset

我们已经定义好了```TCBert```所需的数据格式，你只需要将数据转化为下面的数据格式即可：

### 文本分类
#### 训练数据 示例
```json
{
    "content": "街头偶遇2018款长安CS35，颜值美炸！或售6万起，还买宝骏510？",   
    "label": "汽车"
}

```
#### 标签映射 示例
##### 将真实标签映射为合适的两个字标签
```json
{
    "体育活动":"体育", 
    "军事活动":"军事", 
    "农业":"农业", 
    "国际新闻":"国际", 
    "娱乐新闻":"娱乐", 
    "汽车":"汽车"
}
```

## License

[Apache License 2.0](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/LICENSE)


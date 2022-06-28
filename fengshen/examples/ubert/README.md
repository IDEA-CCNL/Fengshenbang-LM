# Ubert: -------
- 论文[]()
- 知乎[]()

### 简介
Ubert 是我们在做 [2022AIWIN 世界人工智能创新大赛：中文保险小样本多任务](http://ailab.aiwin.org.cn/competitions/68#results) 时提出的一种解决方案。并取得A/B榜榜首的成绩，且B榜综合成绩领先第二名超过 1 个百分点，领先第三名接近 5 个百分点。相比于官方提供的 baseline，提高 20 个百分点。Ubert 不仅可以完成 实体识别、事件抽取等常见抽取任务，还可以完成新闻分类、自然语言推理等分类任务，且所有任务是共享一个统一框架、统一任务、统一训练目标的模型。解题思路和方案可以参考我们的答辩PPT，或者参考我们的知乎文章

## 开源模型列表
 开源的模型是我们在比赛模型的基础上重新整理 70+ 份数据，共 100万+条样本，进行预训练而得到的，可直接开箱即用。开源模型地址如下：
| 模型 | 地址   |
|:---------:|:--------------:|
| Erlangshen-Ubert-110M  | [https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-110M](https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-110M)       |
| Erlangshen-Ubert-330M  | [https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-330M](https://huggingface.co/IDEA-CCNL/Erlangshen-Ubert-330M)   |


## 快速开箱使用
安装我们的 fengshen 框架，我们暂且提供如下方式安装
```python
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable ./
```

一键运行下面代码得到预测结果, 你可以任意修改示例 text 和要抽取的 entity_type，体验一下 Zero-Shot 性能
```python
import argparse
from fengshen import UbertPiplines

total_parser = argparse.ArgumentParser("TASK NAME")
total_parser = UbertPiplines.piplines_args(total_parser)
args = total_parser.parse_args()

test_data=[
    {
        "task_type": "抽取任务", 
        "subtask_type": "实体识别", 
        "text": "这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。", 
        "choices": [ 
            {"entity_type": "小区名字"}, 
            {"entity_type": "岗位职责"}
            ],
        "id": 0}
]

model = UbertPiplines(args)
result = model.predict(test_data)
for line in result:
    print(line)
```

## 继续 finetune 使用

开源的模型我们已经经过大量的数据进行预训练而得到，可以直接进行 Zero-Shot，如果你还想继续finetune,可以参考我们的 [example.py](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/ubert/example.py)。你只需要将我们数据预处理成为我们定义的格式，即可使用简单的几行代码完成模型的训练和推理。我们是复用 pytorch-lightning 的 trainer 。在训练时，可以直接传入 trainer 的参数，此外我们还定义了一些其他参数。常用的参数如下：


```sh
--pretrained_model_path       #预训练模型的路径，默认
--load_checkpoints_path       #加载模型的路径，如果你finetune完，想加载模型进行预测可以传入这个参数
--batchsize                   #批次大小, 默认 8
--monitor                     #保存模型需要监控的变量，例如我们可监控 val_span_acc
--checkpoint_path             #模型保存的路径, 默认 ./checkpoint
--save_top_k                  #最多保存几个模型, 默认 3
--every_n_train_steps         #多少步保存一次模型, 默认 100
--learning_rate               #学习率, 默认 2e-5
--warmup                      #预热的概率, 默认 0.01
--default_root_dir            #模型日子默认输出路径
--gradient_clip_val           #梯度截断， 默认 0.25
--gpus                        #gpu 的数量
--check_val_every_n_epoch     #多少次验证一次， 默认 100
--max_epochs                  #多少个 epochs， 默认 5
--max_length                  #句子最大长度， 默认 512
--num_labels                  #训练每条样本最多取多少个label，超过则进行随机采样负样本， 默认 10
```

## 数据预处理示例

整个模型的 Piplines 我们已经写好，所以为了方便，我们定义了数据格式。目前我们在预训练中主要含有一下几种任务类型

| task_type | subtask_type   |
|:---------:|:--------------:|
| 分类任务  | 文本分类       |
|           | 自然语言推理   |
|           | 情感分析       |
|           | 多项式阅读理解 |
| 抽取任务  | 实体识别       |
|           | 事件抽取       |
|           | 抽取式阅读理解 |
|           | 关系抽取       |

### 分类任务

#### 普通分类任务
对于分类任务，我们把类别描述当作是 entity_type，我们主要关注 label 字段，label为 1 表示该该标签是正确的标签。如下面示例所示
```json
{
	"task_type": "分类任务",
	"subtask_type": "文本分类",
	"text": "7000亿美元救市方案将成期市毒药",
	"choices": [{
		"entity_type": "一则股票新闻",
		"label": 1,
		"entity_list": []
	}, {
		"entity_type": "一则教育新闻",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "一则科学新闻",
		"label": 0,
		"entity_list": []
	}],
	"id": 0
}

```

#### 自然语言推理
```json
{
	"task_type": "分类任务",
	"subtask_type": "自然语言推理",
	"text": "在白云的蓝天下，一个孩子伸手摸着停在草地上的一架飞机的螺旋桨。",
	"choices": [{
		"entity_type": "可以推断出：一个孩子正伸手摸飞机的螺旋桨。",
		"label": 1,
		"entity_list": []
	}, {
		"entity_type": "不能推断出：一个孩子正伸手摸飞机的螺旋桨。",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "很难推断出：一个孩子正伸手摸飞机的螺旋桨。",
		"label": 0,
		"entity_list": []
	}],
	"id": 0
}
```


#### 语义匹配

```json
{
	"task_type": "分类任务",
	"subtask_type": "语义匹配",
	"text": "不要借了我是试试看能否操作的",
	"choices": [{
		"entity_type": "不能理解为：借款审核期间能否取消借款",
		"label": 1,
		"entity_list": []
	}, {
		"entity_type": "可以理解为：借款审核期间能否取消借款",
		"label": 0,
		"entity_list": []
	}],
	"id": 0
}

```

### 抽取任务
对于抽取任务，label 字段是无效的
#### 实体识别
```json
{
	"task_type": "抽取任务",
	"subtask_type": "实体识别",
	"text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，",
	"choices": [{
		"entity_type": "地址",
		"label": 0,
		"entity_list": [{
			"entity_name": "台湾",
			"entity_type": "地址",
			"entity_idx": [
				[15, 16]
			]
		}]
	}{
		"entity_type": "政府机构",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "电影名称",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "人物姓名",
		"label": 0,
		"entity_list": [{
			"entity_name": "彭小军",
			"entity_type": "人物姓名",
			"entity_idx": [
				[0, 2]
			]
		}]
	},
	"id": 0
}

```
#### 事件抽取
```json

{
	"task_type": "抽取任务",
	"subtask_type": "事件抽取",
	"text": "小米9价格首降，6GB+128GB跌了200，却不如红米新机值得买",
	"choices": [{
		"entity_type": "降价的时间",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "降价的降价方",
		"label": 0,
		"entity_list": []
	}, {
		"entity_type": "降价的降价物",
		"label": 0,
		"entity_list": [{
			"entity_name": "小米9",
			"entity_type": "降价的降价物",
			"entity_idx": [
				[0, 2]
			]
		}, {
			"entity_name": "小米9",
			"entity_type": "降价的降价物",
			"entity_idx": [
				[0, 2]
			]
		}]
	}, {
		"entity_type": "降价的降价幅度",
		"label": 0,
		"entity_list": []
	}],
	"id": 0
}
```
#### 抽取式阅读理解

```json
{
	"task_type": "抽取任务",
	"subtask_type": "抽取式阅读理解",
	"text": "截至2014年7月1日，圣地亚哥人口估计为1381069人，是美国第八大城市，加利福尼亚州第二大城市。它是圣迭戈-蒂华纳城市群的一部分，是美国与底特律-温莎之后的第二大跨境城市群，人口4922723。圣地亚哥是加州的出生地，以全年温和的气候、天然的深水港、广阔的海滩、与美国海军的长期联系以及最近作为医疗和生物技术发展中心而闻名。",
	"choices": [{
		"entity_type": "除了医疗保健，圣迭戈哪个就业部门已经强势崛起？",
		"label": 0,
		"entity_list": [{
			"entity_name": "生物技术发展",
			"entity_idx": [
				[153, 158]
			]
		}]
	}, {
		"entity_type": "在所有的军事部门中，哪一个在圣地亚哥的存在最为强大？",
		"label": 0,
		"entity_list": [{
			"entity_name": "美国海军",
			"entity_idx": [
				[135, 138]
			]
		}]
	}, {
		"entity_type": "在美国十大城市中，圣迭戈排名哪一位？",
		"label": 0,
		"entity_list": [{
			"entity_name": "第八",
			"entity_idx": [
				[33, 34]
			]
		}]
	}],
	"id": 0
}
```


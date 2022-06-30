# 二郎神打CLUE语义匹配榜
  - [比赛介绍](#比赛介绍)
  - [clue语义匹配榜打榜思路](#clue语义匹配榜-打榜思路)
  - [数据集介绍](#数据集介绍)
  - [环境](#环境)
  - [用法](#用法)
  - [提交](#提交)

## 比赛介绍
- clue的语义匹配榜 (https://www.cluebenchmarks.com/sim.html)
- clue sim官方实例 (https://github.com/CLUEbenchmark/QBQTC)

## clue语义匹配榜 打榜思路

- 直接使用fengshenbang的二郎神模型，就打到了前三。
- 为了解决标签平衡问题，设计了一个交叉熵平滑滤波loss，就达到了第一。

## 数据集介绍

QQ浏览器搜索相关性数据集（QBQTC,QQ Browser Query Title Corpus），是QQ浏览器搜索引擎目前针对大搜场景构建的一个融合了相关性、权威性、内容质量、
时效性等维度标注的学习排序（LTR）数据集，广泛应用在搜索引擎业务场景中。

相关性的含义：0，相关程度差；1，有一定相关性；2，非常相关。数字越大相关性越高。

**数据量统计**

| 训练集（train) | 验证集（dev) | 公开测试集（test_public) | 私有测试集(test) |
| :----: | :----: | :----: | :----: |
| 180,000| 20,000| 5,000 | >=10,0000|

**评测指标**

f1_score来自于sklearn.metrics，计算公式如下：
`F1 =  2 * (precision * recall) / (precision + recall)`

## 环境
* Python >= 3.6
* torch == 1.8.0+cu111
* transforms == 4.6.0
* pytorch-lightning == 1.3.2
* 一张GPU: A100 40G

## 用法

fengshenbang的二郎神模型的使用是非常简单的。

该example下的代码和思想继承自<a href="https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/hf-ds/fengshen/examples/classification/finetune_classification.py">fengshen/examples/classification/finetune_classification.py</a>

如果需要直接使用该python脚本，把官方的数据集处理成如下形式：

```json
{"sentence1": "应届生实习", "sentence2": "实习生招聘-应届生求职网", "label": "1", "id": 0}
```

然后修改其中的<a href="https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/hf-ds/fengshen/examples/classification/finetune_classification.sh">fengshen/examples/classification/finetune_classification.sh</a>的参数即可。

下面介绍该example的用法：

### 创建文件夹

- dataset 文件夹，下载官方数据集后放进来就行
- weights 文件夹，用以存放二郎神模型
- submissions 文件夹，用以存放需要评测的json文件

### Train
```bash
python main.py \
    --mode 'Train' \
    --model_path './weights/Erlangshen-MegatronBert-1.3B-Similarity' \
    --model_name 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity'
```

加载最优的模型用以Test set的预测。

### Test
```bash
python main.py \
    --mode 'Test' \
    --predict_model_path 'your_model_path' \
    --model_path './weights/Erlangshen-MegatronBert-1.3B-Similarity' \
    --model_name 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity'
```

## 提交

在路径 ./submissions 下，找到 qbqtc_predict.json 并且提交到<a href="https://www.CLUEbenchmarks.com">测评系统</a>

注意：名字必须为qbqtc_predict.json
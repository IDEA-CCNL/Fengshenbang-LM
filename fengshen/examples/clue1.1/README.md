# 中文 NLP 权威测评基准 CLUE 刷榜 Top10 方案指南

 [CLUE](https://www.cluebenchmarks.com) 是中文 NLP 的权威测评榜单，也吸引了许多国内许多团队在上面进行测评。在我们的最新模型 UniMC 中，也使用 CLUE 对我们的模型进行了测评。在全量数据榜单 CLUE1.1 中，我们的 [UniMC-DeBERTa-1.4B](https://huggingface.co/IDEA-CCNL/Erlangshen-UniMC-DeBERTa-v2-1.4B-Chinese) 模型取得了第 8 的成绩，是 [CLUE1.1](https://www.cluebenchmarks.com/rank.html) 排行榜(2022年11月14日)前 10 名中唯一开源模型权重和刷榜代码的模型。

## 刷榜方案

通过观察可以发现，在CLUE需要测评的 9 个任务中，有 8 个是分类任务，只有一个 cmrc2018 是抽取式的阅读理解任务。因此，结合我们的 Fengshenbang-LM 已有的模型，我们可以使用 [UniMC](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/dev/yangping/fengshen/examples/unimc) 来实现 8 个是分类任务，用 [Ubert](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/dev/yangping/fengshen/examples/ubert) 来实现 cmrc2018 任务，详细的方案可以看我们的知乎文章：https://zhuanlan.zhihu.com/p/583679722

## 项目要求

安装我们的 fengshen 框架，我们暂且提供如下方式安装
```shell
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable ./
```
## 运行项目

### 数据下载
由于 HuggingFace 上的数据与最终提交的数据 id 有可能对应不上，所以建议还是去官方仓库进行下载
https://github.com/CLUEBENCHMARK/CLUE


### 数据预处理
将数据下载之后，修改下面脚本的路径，运行下面脚本将数据处理成 UniMC 模型 和 Ubert 模型所需要的格式
```shell
sh cluedata2unidata.sh
```

### 模型训练
训练CLUE上的8个分类任务，一些训练参数可根据自己的设备进行修改。对于全量数据来说，训练超参数没有那么大的影响
```shell
sh run_clue_unimc.sh
```
训练 cmrc2018 任务，一些训练参数可根据自己的设备进行修改
```shell 
sh run_clue_ubert.sh
```

### 预测结果提交

运行下面脚本将预测结果转化为 CLUE 要求的格式，数据路径需要根据自己的路径修改调整。运行下面脚本就可以得到结果，然后拿到 [CLUE](https://www.cluebenchmarks.com/index.html) 官网上去提交了

```shell
sh predict2submit.sh
```



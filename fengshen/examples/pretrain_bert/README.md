# Bert预训练

## 背景

我们有持续收集了一部分语料，有一套自建的数据处理流程。位了验证数据处理的效果，从零开始预训练了2个base级别的Bert模型，一个是基于自建数据，一个是基于同行们开源的数据。总体来说数据效果差别不大，下面只介绍一下本次预训练的流程。

## 数据处理

我们的原始语料主要源自common crawl以及一些开源的高质量语料，经过一些列的数据清洗之后，我们的数据格式为jsonline。例如（摘自内部数据）：
```json
{"text":"据悉,河南博物馆成立于1927年,目前拥有超过170000件(套)的文物收藏,包括Jiahu骨笛,雌性猫头鹰雕像,cloud-patterned铜禁,Duling Fangding,莲花和起重机广场,和玉柄剑,黄金从武则天滑落,四神云雾壁画和汝窑天蓝釉雕鹅颈瓶是九大镇厅的珍品。院中的藏品以史前文物、商周青铜器、陶瓷、玉器和石雕等为特色。高质量文物数量多、品种齐全、品位高、价值高。它们是见证中国文明发展、展示中国历史发展的文化艺术宝库。"}
{"text": "功夫不负有心人，1925年，万氏兄弟试制动画片初获成果，并获得了商务印书馆的大力支持。其后兄弟们再接再厉，直到1927年，一部黑白无声动画片《大闹画室》诞生了爱尔兰风笛。据《申报》记载，“该片内容画人与真人合作锁梦楼，滑稽处甚多，令人观后，捧腹不止。”此片曾远销美国放映，并大受赞誉。1930年夏俊娜，万古蟾到大中华百合影片公司工作，万氏兄弟采用了同样的手法拍摄了第二部动画短片《纸人捣乱记》，并于1931年上映。"}
```

处理脚本路径：`/cognitive_comp/wuziwei/codes/Fengshenbang-LM/fengshen/data/bert_dataloader`

该路径下面有3个文件，`auto_split.sh`和`preprocessing.py`是原始数据预处理的脚本，`load.py是fs_data`的处理脚本，执行顺序如下：

#### step 1

执行`auto_split.sh`文件，作用是分割大文件，超过1GB的文件，会自动分割未300M的小文件。使用方法如下：

`sh auto_split.sh 你的数据文件路径`

#### step 2

执行`preprocessing.py`文件，该文件的作用主要是分句，为什么不嵌入到collate_fn中做，是发现那样效率会慢一些，所以单独拿出来做了。
执行`python preprocessing.py`即可，注意修改脚本内的文件路径。

#### step 3

`load.py`文件是用fsdata的方式加载数据集，也是执行即可。执行一遍，后续的加载可以实现180GB的数据秒入～

前面两步是为了提高load.py文件生成缓存文件的速度。经过这几步的处理以及collate_fn函数（bert mask 策略的实现），最终变成bert的输入。如下：

*ps: collate_fn在`Fengshenbang-LM\fengshen\examples\pretrain_bert\pretrain_bert.py`脚本下，由DataCollate类实现。*

```json
{
"input_ids": torch.tensor(input_ids),
"labels": torch.tensor(batch_labels),
"attention_mask": torch.tensor(attention_mask),
"token_type_ids": torch.tensor(token_type_ids)
}
```

## 模型结构

模型结构即为标准的bert-base，即：
|    配置     | 参数  |
| :---------: | :---: |
|   nlayers   |  12   |
|  nheaders   |  12   |
| hidden-size | 768  |
| seq-length  | 512  |
| vocab-size  | 21128  |

## 任务以及Mask策略

*mask策略的实现在`Fengshenbang-LM\fengshen\examples\pretrain_bert\pretrain_bert.py`的**DataCollate**类中*

本次预训练取消了NSP任务，只做mask任务，具体mask策略如下：

- 15%随机mask
    - 80% mask
    - 10% 随机替换
    - 10% 保持不变
- 全词mask （wwm）
- n-gram mask

由于加入了全词mask和n-gram mask 总体的mask token数量会比英文原始论文的mask比例略高

## 预训练执行流程

- 训练框架：[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- 脚本执行：`sh Fengshenbang-LM\fengshen\examples\pretrain_bert\pretrain_bert.sh`

*具体配置见`Fengshenbang-LM\fengshen\examples\pretrain_bert\pretrain_bert.sh`*

## Randeng-BART-139M-QG-Chinese



## 简介 Brief Introduction

善于处理问题生成任务的中文版 BART-base 模型。

Good at solving question generation tasks Bart-base Model (Chinese version).

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | BART |      139M      |    问题生成任务-中文 QuestionGeneration-Chinese    |


## 模型信息 Model Information

本模型基于[IDEA-CCNL/Randeng-BART-139M](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M)，我们在 [ChineseSQuAD](https://github.com/pluto-junzeng/ChineseSquad) 数据集上微调了问题生成任务版本。

Based on [IDEA-CCNL/Randeng-BART-139M](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M), we fine-tuned a question generation version on [ChineseSQuAD](https://github.com/pluto-junzeng/ChineseSquad) datasets.


Table1: 模型结构和配置 Model Architecture and Config

|    配置 Config      |  参数 Value|
| ------------------- | --------- |
| encoder layers      |     6    |
| encoder_attn_heads  |     12    |
| encoder_ffn_dim     |     3072  |
| decoder_layers      |     6    |
| decoder_attn_heads  |     12    |
| decoder_ffn_dim     |    3072   |
| max_encoder_len     |    512    |


ChineseSQuAD 数据集翻译了部分SQuAD数据集，包含约 67k 有答案的训练样本和 43k 无答案训练样本。我们做了 9:1 的训练-开发集合划分，并在公开的开发集上评测了效果。

The dataset is translated from SQuAD 2.0, with around 67k samples with answers and 43k samples without answers. We split the data to train-dev with ratio of 9:1 and test the performance on the public dev set.

Table 2: 数据集样本量
|       | all    | have ans | no ans |
|:------|:-------|:---------|:-------|
| train_split | 100097 |    60879 |  39128 |
| dev_split   |  11089 |     6809 |   4280 |
| dev   |  10836 |     6645 |   4191 |


## 使用 Usage

主要文件 key file
- finetune_bart.py 定义了数据处理输入输出方式和finetune的核心代码
- finetune_bart.sh 训练脚本，具体参数可在此修改
- utils.py 定义了独立的工具代码，重实现的函数等

训练 train
安装好 fengshen 所需环境后
```python
bash finetune_bart.sh
```

推理 inference 
```python
from transformers import AutoTokenizer, BartForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Randeng-BART-139M-QG-Chinese",additional_special_tokens=["<ans>"])
model = BartForConditionalGeneration.from_pretrained("IDEA-CCNL/Randeng-BART-139M-QG-Chinese")

context = "知识：1939年9月1日德国入侵波兰后，第二次世界大战开始，华沙一直被保卫到9月27日。波兰中部，包括华沙，都在德国纳粹殖民地政府总政府的统治下。所有的高等教育机构都立即关闭，华沙的犹太人口——几十万，约占城市的 <ans> ——全部涌入华沙的贫民区。回答：30%"
inputs = tokenizer.encode_plus(
            context,
            max_length=448,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
out = model.generate(                
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=True,
        num_beams=5,
        max_length=64,
        top_p = 0.9,
    )
print(pred = tokenizer.batch_decode(out,clean_up_tokenization_spaces=True, skip_special_tokens=True)[0])
# 问题:华沙的犹太人口占城市的百分之多少?
```

### 下游效果 Performance
| Dataset          |  Size  | BLEU-4 | METEOR | ROUGE-L| 
| ------------ | -----  | -------- |--------- | ---------- |
|   ChineseSQuAD               |  139M   |  22.17 |   40.38  |   38.17   | 

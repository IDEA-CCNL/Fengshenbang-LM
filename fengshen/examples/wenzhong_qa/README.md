<style>
table {
margin: auto;
/* width:50%; */
/* cellspacing:20%; */
}
</style>
# <center> yuyuanQA模型finetune
### 数据和模型
#### 模型：
    finetune的模型是yuyuan模型，余元模型是GPT2的结构，在预训练阶段主要是用PubMed医疗相关的数据集进行的预训练。是一个医疗领域的大模型。模型共有35亿参数，主要参数如下表所示：
|||
| ----------- | ---- |
| nlayers     | 30   |
| nheaders    | 32   |
| hidden-size | 3072 |
| seq-length  | 1024 |
    预训练的数据，主要医疗相关的论文、杂志期刊等，以英文语料为主。
#### 数据：
    用于finetune的语料是清洗于[MedQuAD](https://github.com/abachaa/MedQuAD)数据集
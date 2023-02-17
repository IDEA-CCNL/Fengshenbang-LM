# 燃灯系列-T5问答模型微调
## 简介 Brief Introduction
 Here are codes for finetuning Randeng-T5-QA-Chinese. The model was pretrained on the Wudao 180G corpus, and finetuned on Chinese SQuAD and CMRC2018 dataset. It can produce a fluent and accurate answer given a passage and question.

这是中文的生成式问答模型[Randeng-T5-QA-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-T5-784M-QA-Chinese)的微调代码。它基于T5-Large结构，使用悟道180G语料在[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)进行预训练，在ChineseSQuAD和CMRC2018两个阅读理解数据集上进行微调。输入一篇文章和一个问题，可以生成准确流畅的回答。

## 模型类别 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | T5 |      784M      |     中文生成式问答 -Chinese Generative Qustion Answering   |

模型架构
| 配置 | 参数 |
| ---- | ---- |
| encoder layers | 12 |
| encoder_attention_heads | 16 |
| encoder_ffn_dim | 2816 |
| decoder layers | 24 |
| decoder_attention_heads| 16 |
| decoder_ffn_dim | 2816 |
| max_encode_length | 1024 |

## 模型表现 Performance 

 CMRC 2018的测试集上的效果（原始任务是一个起始和结束预测问题，这里作为一个生成回答的问题）
  
   | model | Contain Answer Rate| RougeL | BLEU-4 |F1 | EM | 
   |-------|----|----|--------------------|--------|--------|
   | Ours | 76.0 | 82.7 |61.1|77.9 |57.1|
  
   
   Our model enjoys a high level of generation quality and accuracy, with 76% of generated answers containing the ground truth. The high RougeL and BLEU-4 reveal the overlap between generated results and ground truth. Our model has a lower EM because it generates complete sentences while golden answers are segmentations of sentences. 

   我们的模型有着极高的生成质量和准确率，76%的回答包含了正确答案(Contain Answer Rate)。RougeL和BLEU-4反映了模型预测结果和标准答案重合的程度。我们的模型EM值较低，因为生成的大部分为完整的句子，而标准答案通常是句子片段。


## 模型

T5-Large: [Randeng-T5-784M-QA-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-T5-784M-QA-Chinese)

文件：
 - qa_dataset.py 数据集的处理，包含dataset和dataloader
 - finetune_t5_cmrc.py 模型微调核心代码
 - run_finetune.sh, 微调脚本(未安装deepspeed的话strategy参数改为ddp)
 - run_predict2.sh 预测脚本

## 使用 Usage

```python
import numpy as np
from transformers import T5Tokenizer,MT5ForConditionalGeneration

pretrain_path = 'IDEA-CCNL/Randeng-T5-784M-QA-Chinese'
tokenizer=T5Tokenizer.from_pretrained(pretrain_path)
model=MT5ForConditionalGeneration.from_pretrained(pretrain_path)

max_knowledge_length = 425
max_new_tokens = 128
max_seq_length = 512

sample={"context":"在柏林,胡格诺派教徒创建了两个新的社区:多罗西恩斯塔特和弗里德里希斯塔特。到1700年,这个城市五分之一的人口讲法语。柏林胡格诺派在他们的教堂服务中保留了将近一个世纪的法语。他们最终决定改用德语,以抗议1806-1807年拿破仑占领普鲁士。他们的许多后代都有显赫的地位。成立了几个教会,如弗雷德里夏(丹麦)、柏林、斯德哥尔摩、汉堡、法兰克福、赫尔辛基和埃姆登的教会。","question":"除了多罗西恩斯塔特,柏林还有哪个新的社区?","idx":1}
plain_text='question:'+sample['question']+'knowledge:'+sample['context'][:max_knowledge_length]

res_prefix=tokenizer.encode('answer'+'<extra_id_0></s>',add_special_tokens=False)
l_rp=len(res_prefix)

tokenized=tokenizer.encode(plain_text,add_special_tokens=False,truncation=True,max_length=max_seq_length-2-l_rp)
tokenized+=res_prefix

## generate answer
input_ids=torch.tensor([tokenized])
pred_ids = model.generate(input_ids=input_ids,max_new_tokens=max_new_tokens,do_sample=True,top_p=0.9)
tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

## 引用 Citation
如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2210.08590)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2210.08590):

```text
@article{fengshenbang,
  author    = {Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen and Ruyi Gan and Jiaxing Zhang},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

欢迎引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):
```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```

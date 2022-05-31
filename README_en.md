[**中文**](./README.md) | [**English**](./README_en.md)

# Navigation
- [Navigation](#navigation)
- [Intro](#intro)
- [Fengshenbang-LM](#fengshenbang-lm)
  - [Erlangshen](#erlangshen)
    - [Download the Models](#download-the-models)
    - [Load the Models](#load-the-models)
    - [Example of Usage](#example-of-usage)
    - [Performance on Downstream Tasks](#performance-on-downstream-tasks)
- [Fengshen Framework](#fengshen-framework)
- [A Series of Articles](#a-series-of-articles)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

# Intro
|Model|Scale|Architecture|Domain|Target Tasks|Notes|
|-|-|-|-|-|-|
|[Erlangshen](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E4%BA%8C%E9%83%8E%E7%A5%9E%E7%B3%BB%E5%88%97/index.html)|0.1B-1.3B params|Bidirectional Laugnage Models with Encoder structure|Common|NLU|Largest Chinese open source Bert model; SOTA on few-shot learning benchmark FewCLUE|
|[Wenzhong](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E9%97%BB%E4%BB%B2%E7%B3%BB%E5%88%97/index.html)|0.1B-3.5B params|Unidirectional Language Models with Decoder structure|Common|NLG||
|[Randeng](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E7%87%83%E7%81%AF%E7%B3%BB%E5%88%97/index.html)|770M-0.7B params|Encoder-Decoder structured models with Transformer/T5 structures|Common|NLU+NLG||
|[Yuyuan](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E4%BD%99%E5%85%83%E7%B3%BB%E5%88%97/index.html)|0.1B-3.5B params|Unidirectional Language Models with GPT-2 structure|Medical|NLG|Largest open source GPT2 medical model|
|[Bigan](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E6%AF%94%E5%B9%B2%E7%B3%BB%E5%88%97/index.html)|1.1B params| Transformer-XL structures|Common|Semantic Correction | |
|[Zhouwenwang](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%91%A8%E6%96%87%E7%8E%8B%E7%B3%BB%E5%88%97/index.html)|0.1B-1.3B params|Unified Language Models|Common|NLU+NLG|Modified based on Roformer structure; the largest model trained on both LM and MLM|
|[Taiyi](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%A4%AA%E4%B9%99%E7%B3%BB%E5%88%97/index.html)|87M-0.1B params| Co-attention | Multimodal | Multimodal Semantic understanding | |


[Download url of Fengshenbang](https://huggingface.co/IDEA-CCNL)

[Handbook of Fengshenbang](https://fengshenbang-doc.readthedocs.io/zh/latest/index.html)


# Fengshenbang-LM
Pretraining of large-scale models have gradually become the basis of cognitive computing in recent years; Tasks and algorithms of natural language processing and computer vision heavily rely on pretrained large models.

The scale of pretrained models, measured by the number of parameters, have been growing at a rate of approximately 10 times per year from the initial 110M BERT to the 175B GPT-3. For different downstream tasks, pretrained models that vary in architectures, sizes and expertise domains are needed; in short, the world needs more larger models. 

However, limited computing power is a major bottleneck for the development of the field. Institutions such as most universities, small companies and businesses in traditional areas are not equipped with enough computing resources for training and inferencing with large-scale pretrained models; further industrial practice of artificial intelligence is hence hindered.

And the world needs an answer for this.

IDEA (International Digital Economy Academy) officially announces the launch of "Fengshenbang" open source project. It open sources a series of large-scale natural languguage pretrained models. These models will bring comprehensive coverage across various model architectures, sizes and expertise domains. We guarantee that we will optimize the models continuously with new datasets and latest algorithms. We aim to build universal infrastructure for Chinese cognitive intelligence and prevent duplicative construction, and hence save computing resources for the community.

![avatar](pics/pic1_eng.png)

We also call for businesses, universities and institutions to join us with the project and build the sytem of large-scale open-source models collaboratively. We envision that, in the near future, the first choice when in need of a new pretrained model should be selecting one in closest proximity to the desired scale,architecture and domain from the series, followed by further training. After obtaining a trained new model, we shall add it back to the series of open-source models for future usage. In this way we build the open-source system iteratively and collaboratively while individuals could get desired models using minimal computing resources. 

![avatar](pics/pic2_eng.png)

For better open source experience, all models of the Fengshenbang series are synchronized within the Huggingface community, and can be obtained for use within few lines of code. Welcome to download and use our models from our repo at [IDEA-CCNL at HuggingFace](https://huggingface.co/IDEA-CCNL).

 
## Erlangshen

This series focuses on using bidirectional language models with encoders to solve multiple natural language understanding tasks. 
Erlangshen-MegatronBert-1.3B is the largest Chinese open source model with the structure of Bert. It contains 13 billion parameters, and was trained with 280G datasets on 32 A100 GPUs for 14 days. It achieved the top on the Chinese natural language understanding benchmark FewCLUE on Nov 10th, 2021. Among the tasks of FewCLUE, Erlangshen-1.3 beat human performance on the task of CHID(Chinese idioms cloze test) and TNEWS(News Classification), and achieved SOTA on tasks of CHID, CSLDCP（academic literature classification) and OCNLI(Natural language Inference), refreshing the records of few-shot learning. We will continue to optimize the Erlangshen series with respect to model scale, knowledge fusion, auxiliary supervision tasks, etc. 

![image](https://user-images.githubusercontent.com/4384420/141752311-d15c2a7f-cd83-4e9e-99a5-cb931088845e.png)

Erlangshen-MRC achieved the Chinese language comprehension evaluations benchmark ZeroCLUE on Jan 24th, 2022. Among the tasks of ZeroCLUE, CSLDCP (discipline literature classification), TNEWS (news classification), IFLYTEK (application description classification), CSL (abstract keyword recognition), CLUEWSC (reference resolution) achieved SOTA.

![image](https://user-images.githubusercontent.com/4384420/151319156-e20ba252-b531-4779-8099-ef60c7954f76.png)

### Download the Models
[Huggingface Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)

### Load the Models 
``` python
from transformers import MegatronBertConfig, MegatronBertModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
config = MegatronBertConfig.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
model = MegatronBertModel.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")

```
### Example of Usage
For the convenience of developers, we offer an example [script](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/scripts/finetune_classification.sh) for downstream finetuning. The script uses the tnews dataset from [CLUE](https://github.com/CLUEbenchmark/CLUE). 

1、Fisrt, modify the MODEL_TYPE and PRETRAINING_MODEL_PATH parameters of [finetune script](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/scripts/finetune_classification.sh), and other parameters can be modified according to your specific device.

``` sh
MODEL_TYPE=huggingface-megatron_bert
PRETRAINED_MODEL_PATH=IDEA-CCNL/Erlangshen-MegatronBert-1.3B
```
2、run

``` sh
sh finetune_classification.sh
```



### Performance on Downstream Tasks 
|     Model   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| Erlangshen-MegatronBert-1.3B | 0.7608      |   0.5996    | 0.6234      |   0.7917    | 0.81    | 0.9243    | 0.872    |


# Fengshen Framework

To facilitate the use of the Fengshenbang large model, further joining in continued training as well as applying in downstream tasks, we synchronously open source the Fengshen framework. Referring to other excellent open source frameworks (including [HuggingFace](https://github.com/huggingface/transformers), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [DeepSpeed](https://github.com/microsoft/DeepSpeed)) and combining the characteristics of NLP field, we redesign FengShen with Pytorch as the base framework and Pytorch-Lightning as the Pipeline. FengShen can be applied to pre-training of large models (tens of billions of parameters) based on massive data (terabytes of data) and fine-tuning on various downstream tasks. Users can easily perform distributed training and memory-saving techniques with configuration, thus focusing more on model implementation and innovation. Also, FengShen can directly use the model structure in [HuggingFace](https://github.com/huggingface/transformers) for continued training, which facilitates domain transfer for users. FengShen provides rich and realistic source code and examples. We will continue to optimize the FengShen framework as the models of Fengshenbang are trained and applied. Stay tuned. 


[Get Started with Fengshen in 3 Minutes](fengshen/README.md)


# A Series of Articles

[Fengshen Series: Getting Started on Training Large Model with Data Parallelism](https://zhuanlan.zhihu.com/p/512194216)

[Fengshen Series: It is Time to Accelerate your Training Process !](https://zhuanlan.zhihu.com/p/485369778)

# Citation
```
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```
# Contact
![avartar](pics/contactus.png)

# License 

[Apache License 2.0](LICENSE)


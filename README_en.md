[**中文**](./README.md) | [**English**](./README_en.md)

# Fengshenbang-LM
Pretraining of large-scale models have gradually become the basis of cognitive computing in recent years; Tasks and algorithms of natural language processing and computer vision heavily rely on pretrained large models.

The scale of pretrained models, measured by the number of parameters, have been growing at a rate of approximately 10 times per year from the initial 110M BERT to the 175B GPT-3. For different downstream tasks, pretrained models that vary in architectures, sizes and expertise domains are needed; in short, the world needs more larger models. 

However, limited computing power is a major bottleneck for the development of the field. Institutions such as most universities, small companies and businesses in traditional areas are not equipped with enough computing resources for training and inferencing with large-scale pretrained models; further industrial practice of artificial intelligence is hence hindered.

And the world needs an answer for this.

IDEA (International Digital Economy Academy) officially announces the launch of "Fengshenbang" open source project. It open sources a series of large-scale natural languguage pretrained models. These models will bring comprehensive coverage across model architectures, sizes, and expertise domains. We guarantee that we will udgrade the models continuously with new datasets and latest algorithms. We aim to build universal infrastructure for Chinese cognitive intelligence and prevent duplicative construction, and hence save computing resources for the community.

![avatar](pic1_eng.png)

We also call for businesses, universities and institutions to join us with the project and build the sytem of large-scale open-source models collaboratively. We envision that, in the near future, the first choice when in need of a new pretrained model should be selecting one in closest proximity to the desired scale,architecture and domain from the series, followed by further training. After obtaining a trained new model, we shall add it back to the series of open-source models for future usage. In this way we build the open-source system iteratively and collaboratively while individuals could get desired models using minimal computing resources. 

![avatar](pic2_eng.png)

For better open source experience, all models of the Fengshenbang series are synchronized within the Huggingface community, and can be obtained for use within few lines of code. Welcome to download and use our models from our repo at [IDEA-CCNL at HuggingFace](https://huggingface.co/IDEA-CCNL).

 
## Erlangshen(二郎神) Series

This series focuses on using bidirectional language models with encoders to solve multiple natural language understanding tasks. 
Erlangshen-1.3B is the largest Chinese open source model with the structure of Bert. It contains 13 billion parameters, and was trained with 280G datasets on 32 A100 GPUs for 14 days. It achieved the top on the Chinese natural language understanding benchmark FewCLUE on Nov 10th, 2021. Among the tasks of FewCLUE, Erlangshen-1.3 beat human performance on the task of CHID(Chinese idioms cloze test) and TNEWS(News Classification), and achieved SOTA on tasks of CHID, CSLDCP（academic literature classification) and OCNLI(Natural language Inference), refreshing the records of few-shot learning. We will continue to optimize the Erlangshen series with respect to model scale, knowledge fusion, auxiliary supervision tasks, etc. 

![image](https://user-images.githubusercontent.com/4384420/141752311-d15c2a7f-cd83-4e9e-99a5-cb931088845e.png)


### Download the Models
[Huggingface Erlangshen-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-1.3B)

### Load the Models 
``` python
from transformers import MegatronBertConfig, MegatronBertModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-1.3B")
config = MegatronBertConfig.from_pretrained("IDEA-CCNL/Erlangshen-1.3B")
model = MegatronBertModel.from_pretrained("IDEA-CCNL/Erlangshen-1.3B")

```
### Example Usage
For the convenience of developers, we offer an example script for downstream finetuning. The script uses the afqmc dataset from [CLUE](https://github.com/CLUEbenchmark/CLUE). DATA_PATH is the path of [afqmc](https://github.com/CLUEbenchmark/CLUE) dataset, and PRETRAINED_MODEL_PATH is the path of the pretrained model. You can download the model from Huggingface to your local directory, and then assign your model saving directory to PRETRAINED_MODEL_PATH. If you don't want to manually download the models, then let PRETRAINED_MODEL_PATH="IDEA-CCNL/Erlangshen-1.3B" and the script will automatically download the models.


``` sh
python example/finetune.py " \
        --train_data_path $TRAIN_DATA_PATH \
        --dev_data_path $DEV_DATA_PATH \
        --test_data_path $TSET_DATA_PATH \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --checkpoints ./model.pth \
        --output_path ./afqmc_predict.json \
        --log_file_path ./finetune.log \
        --batch_size 32 \
        --learning_rate 0.00002 \
        --max_length 64 \
        --epoch 7 \
        --model_type megatron \
            "
```
In order for the developers to do task-adaptive pretraining on the basis of the open-source models, we offer an example script for further pretraining. The script is as follows:

``` sh
python example/pretraining.py " \
        --train_data_path $TRAIN_DATA_PATH \
        --dev_data_path $DEV_DATA_PATH \
        --test_data_path $TSET_DATA_PATH \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --checkpoints ./model.pth \
        --output_path ./afqmc_predict.json \
        --log_file_path ./pretraining.log \
        --batch_size 128 \
        --learning_rate 0.00002 \
        --max_length 64 \
        --epoch 135 \
        --model_type megatron \
            "
```



### Downstream Task Performance
|     Model   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| Erlangshen-1.3B | 0.7608      |   0.5996    | 0.6234      |   0.7917    | 0.81    | 0.9243    | 0.872    |

## Zhouwenwang(周文王) Series
This series, with models of newly-designed architectures, is developed collaboratively by IDEA Cognitive Computing Center and Zhuiyi Technology. The models consider LM(Language Model) and MLM(Masked Language Model) jointly from the pretraining stage, and utilize Rotary Position Embedding, so that the resulting models are capable of both language generation and language understanding. We currently present Zhouwenwang-1.3B model with 1.3 billion parameters; it is the largest Chinese language model that handles LM and MLM tasks at the same time. We will continue to optimize the Zhouwenwang series with respect to model scale, knowledge fusion, auxiliary supervision tasks, etc. 


### Download The Models

[Huggingface Zhouwenwang-1.3B](https://huggingface.co/IDEA-CCNL/Zhouwenwang-1.3B)<br>
[Huggingface Zhouwenwang-110M](https://huggingface.co/IDEA-CCNL/Zhouwenwang-110M)
### Load the Models
Currently our Zhouwenwang series of models are modified based on the Roformer structure from Zhuiyi Technology, and we have not added Zhouwenwang series to Huggingface yet. Therefore for now you need to load the model files from this repo to your own working directory, then you can follow the script below to download corresponding models from Huggingface and import them.

``` python
from model.roformer.modeling_roformer import RoFormerModel            #Import Roformer Model from the Roformer Files in this Repo
from model.roformer.configuration_roformer import RoFormerConfig
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Zhouwenwang-110M')
config = RoFormerConfig.from_pretrained('IDEA-CCNL/Zhouwenwang-110M')
model = RoFormerModel.from_pretrained('IDEA-CCNL/Zhouwenwang-110M')
```


### Example Usage

``` sh
python example/finetune.py " \
        --train_data_path $TRAIN_DATA_PATH \
        --dev_data_path $DEV_DATA_PATH \
        --test_data_path $TSET_DATA_PATH \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --checkpoints ./model.pth \
        --output_path ./afqmc_predict.json \
        --log_file_path ./finetune.log \
        --batch_size 32 \
        --learning_rate 0.00002 \
        --max_length 64 \
        --epoch 7 \
        --model_type roformer \
            "
```

### Downstream Task Performance

#### Natural Language Understanding
When using Zhouwenwang-1.3B for NLU tasks, the token_type should be all set to 0. The performance of Zhouwenwang-1.3B on downstream tasks is as follows:

|    Model   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| Zhouwenwang-1.3B | 0.7463     |   0.6036    | 0.6288     |   0.7654   | 0.7741    | 0.8849    | 0. 8777   |

#### Natural Language Generation
When using Zhouwenwang-1.3B for NGL tasks, the token_type should be all set to 1. The performance of Zhouwenwang-1.3B on downstream tasks is as follows:

```python
from model.roformer.modeling_roformer import RoFormerModel
from transformers import AutoTokenizer
import torch
import numpy as np

sentence = '清华大学位于'
max_length = 32

tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Zhouwenwang-110M')
model = RoFormerModel.from_pretrained('IDEA-CCNL/Zhouwenwang-110M')

for i in range(max_length):
    encode = torch.tensor(
        [[tokenizer.cls_token_id]+tokenizer.encode(sentence, add_special_tokens=False)]).long()
    logits = model(encode)[0]
    logits = torch.nn.functional.linear(
        logits, model.embeddings.word_embeddings.weight)
    logits = torch.nn.functional.softmax(
        logits, dim=-1).cpu().detach().numpy()[0]
    sentence = sentence + \
        tokenizer.decode(int(np.random.choice(logits.shape[1], p=logits[-1])))
    if sentence[-1] == '。':
        break
print(sentence)

 ```



## Wenzhong(闻仲) Series
The Wenzhong Series is a group of powerful generative models that consist of unidirectional lanugage models of decoder structure. 
The Wenzhong-3.5B model is trained with 100G datasets on 256 A100 GPUs for 28 hours, and contains 3.5 billion parameters. 

### Download the Models
[Huggingface Wenzhong-3.5B](https://huggingface.co/IDEA-CCNL/Wenzhong-3.5B)

### Load the Models
```python 
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong-3.5B')
model = GPT2Model.from_pretrained('IDEA-CCNL/Wenzhong-3.5B')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
### Language Generation
```python
from transformers import pipeline, set_seed
set_seed(55)
generator = pipeline('text-generation', model='IDEA-CCNL/Wenzhong-3.5B')
generator("北京是中国的", max_length=30, num_return_sequences=1)

```


## Randeng(燃灯) Series
The Randong Series is a group of encoding-decoding language models of transformer structure. 
The Randing-770M model is trained with 280G datasets on 16 A100 GPUs for 14 days, and contains 770M parameters. 

### Download the Models
[Huggingface Randeng-770M](https://huggingface.co/IDEA-CCNL/Randeng-770M/)

### Load the Models
Our Randeng-770M is trained based on the T5 structure of Megatron. Since the T5 model structure of Megatron is slightly different from the T5 model structure of Huggingface, directly importing Randeng using HuggingFace T5 is not supported. You need to load the model files from this repo to your own working directory, then you can follow the script below to download corresponding models from Huggingface and import them.

``` python
from model.megatron_t5.modeling_megatron_t5 import T5ForConditionalGeneration
from model.megatron_t5.configuration_magetron_t5 import T5Config
from model.megatron_t5.tokenization_megatron_t5 import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('IDEA-CCNL/Randeng-770M')
config = T5Config.from_pretrained('IDEA-CCNL/Randeng-770M')
model = T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-770M')
```

### Example Usage

#### Example for Downstream Task Usage
``` sh
python example/finetune.py " \
        --train_data_path $TRAIN_DATA_PATH \
        --dev_data_path $DEV_DATA_PATH \
        --test_data_path $TSET_DATA_PATH \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --checkpoints ./model.pth \
        --output_path ./afqmc_predict.json \
        --log_file_path ./finetune.log \
        --batch_size 32 \
        --learning_rate 0.00002 \
        --max_length 64 \
        --epoch 7 \
        --model_type megatron_t5 \
            "
```
#### Example for Generation Task

```python
from model.megatron_t5.modeling_megatron_t5 import T5ForConditionalGeneration
from model.megatron_t5.tokenization_megatron_t5 import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('IDEA-CCNL/Randeng-770M')
model = T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-770M')

output = model.generate(tokenizer.encode(tokenizer.encode('北京是中国的<extra_id_0>')))
print(tokenizer.decode(output))

```



## Yuyuan(余元) Series
The Yuyuan series is a group of models focusing on the medical domain. The Yuyuan-3.5B model is trained with 50G medical domain dataset and knowledge based on common domain pretrained models on 32 A100 for 7 days; it is by far the largest open source GPT-2 medical domain language model. It achieves near 90% accuracy in SOP judgement in medical domain. 
We utilize Yuyuan-3.5B model for factual judgement and medical Q&A. We are looking forward to more possibilities from you. 


### Download the Models
[Huggingface Yuyuan-3.5B](https://huggingface.co/IDEA-CCNL/Yuyuan-3.5B)

### Load the Models
```python 
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Yuyuan-3.5B')
model = GPT2Model.from_pretrained('IDEA-CCNL/Yuyuan-3.5B')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
### Language Generation
```python
from transformers import pipeline, set_seed
set_seed(55)
generator = pipeline('text-generation', model='IDEA-CCNL/Yuyuan-3.5B')
generator("Diabetics should not eat", max_length=30, num_return_sequences=1)

```

## Citation
```
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```
## Contact Us
![avartar](contactus.png)

## License

[Apache License 2.0](LICENSE)

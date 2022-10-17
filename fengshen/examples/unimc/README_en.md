[**中文**](./README.md) | [**English**](./README_en.md)
# UniMC
Code for  [Zero-Shot Learners for Natural Language Understanding via a Unified Multiple Choice Perspective]()



![](./unimc.jpg)

## Update
- [2022-10-16] Release preprint in arXiv.
- [2022-10-14] Release code in GitHub.

## Requirements


```shell
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable .
```

## Quick Start
You can refer to our [example.py]()

```python
import argparse
from fengshen.pipelines.multiplechoice import UniMCPiplines

total_parser = argparse.ArgumentParser("TASK NAME")
total_parser = UniMCPiplines.piplines_args(total_parser)
args = total_parser.parse_args()
    
pretrained_model_path = 'IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English'
args.language='english'
args.learning_rate=2e-5
args.max_length=512
args.max_epochs=3
args.batchsize=8
args.default_root_dir='./'
model = UniMCPiplines(args, model_path=pretrained_model_path)

train_data = [] 
dev_data = [] 
test_data = [{
	"texta": "it 's just incredibly dull .",
	"textb": "",
	"question": "What is sentiment of follow review?",
	"choice": ["it's great", "it's terrible"],
	"answer": "",
	"label": 0,
	"id": 19
}]

if args.train:
	model.fit(train_data, dev_data)
result = model.predict(test_data)
```
## Pretrained Model
For the English model, the model was pre-trained with 14 multiplechoice datasets. For the Chinese model, we have collected 48 datasets to pre-train the model, and we have open sourced the pre-trained model to the HuggingFace community.

| Model | URL   |
|:---------:|:--------------:|
| Erlangshen-Albert-235M-UniMC-English  | [https://huggingface.co/IDEA-CCNL/Erlangshen-Albert-235M-UniMC-English](https://huggingface.co/IDEA-CCNL/Erlangshen-Albert-235M-UniMC-English)   |
| Erlangshen-RoBERTa-110M-UniMC-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-RoBERTa-110M-UniMC-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-RoBERTa-110M-UniMC-Chinese)       |
| Erlangshen-RoBERTa-330M-UniMC-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-RoBERTa-330M-UniMC-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-RoBERTa-330M-UniMC-Chinese)   |
| Erlangshen-MegatronBERT-1.3B-UniMC-Chinese  | [https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBERT-1.3B-UniMC-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBERT-1.3B-UniMC-Chinese)       |


## Experiments
To evaluate the performance of UniMC, we use 14 multiple-choice datasets to pre-train the model with the ability to make choices

**Zero-shot**
| Model   | T0 11B | GLaM 60B | FLAN 137B | PaLM 540B | UniMC 235M |
|---------|--------|----------|-----------|-----------|------------|
| ANLI R1 | 43.6   | 40.9     | 47.7      | 48.4      | **52.0**         |
| ANLI R2 | 38.7   | 38.2     | 43.9      | 44.2      | **44.4**       |
| ANLI R3 | 41.3   | 40.9     | 47.0        | 45.7      | **47.8**       |
| CB      | 70.1   | 33.9     | 64.1      | 51.8      | **75.7**       |

## Citation
If this repository helps you, please cite this paper:

```text
@article{DBLP:journals/corr/abs-2209-02970,
  author    = {Junjie Wang and
               Yuxiang Zhang and
               Lin Zhang and
               Ping Yang and
               Xinyu Gao and
               Ziwei Wu and
               Xiaoqun Dong and
               Junqing He and
               Jianheng Zhuo and
               Qi Yang and
               Yongfeng Huang and
               Xiayu Li and
               Yanghan Wu and
               Junyu Lu and
               Xinyu Zhu and
               Weifeng Chen and
               Ting Han and
               Kunhao Pan and
               Rui Wang and
               Hao Wang and
               Xiaojun Wu and
               Zhongshen Zeng and
               Chongpei Chen and
               Ruyi Gan and
               Jiaxing Zhang},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

## License

[Apache License 2.0](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/LICENSE)


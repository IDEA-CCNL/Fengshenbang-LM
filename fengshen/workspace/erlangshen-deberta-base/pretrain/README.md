---
language: 
  - zh

license: apache-2.0

tags:
  - bert

inference: true

widget:
- text: "生活的真谛是[MASK]。"
---
# Erlangshen-Deberta-97M-Chinese，one model of [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM).
The 97 million parameter deberta-V2 base model, using 180G Chinese data, 24 A100(40G) training for 7 days，which is a encoder-only transformer structure. Consumed totally 1B samples.


## Task Description

Erlangshen-Deberta-97M-Chinese is pre-trained by bert like mask task from Deberta [paper](https://readpaper.com/paper/3033187248)


## Usage
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese', use_fast=False)
model=AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese')
text = '生活的真谛是[MASK]。'
fillmask_pipe = FillMaskPipeline(model, tokenizer, device=7)
print(fillmask_pipe(text, top_k=10))
```

## Finetune

We present the dev results on some tasks.

| Model                              | OCNLI | CMNLI  |
| ---------------------------------- | ----- | ------ |
| RoBERTa-base                       | 0.743 | 0.7973 |
| **Erlangshen-Deberta-97M-Chinese** | 0.752 | 0.807  |

## Citation
If you find the resource is useful, please cite the following website in your paper.
```
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2022},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```
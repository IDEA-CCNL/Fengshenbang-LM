# translation examples
## 数据预处理

数据预处理部分目前整合不多，主要提供一个最终的转换文件，转换成模型应用格式，前期还是应用mose等工具进行数据的预处理部分，产出处理后的源目标语言和目标语言两份数据，再调用本脚本合并

前期数据预处理脚本可参考，deltalm在fairseq中的demo，prepare_iwslt14.sh：https://github.com/microsoft/unilm/blob/master/deltalm/examples/prepare_iwslt14.sh)

### 目标格式
需要将翻译的源语言和目标语言转换到一个文件中，格式如下：
src为源语言，tgt为目标语言，每一行都是一个json格式
```
{"src": "und was menschliche gesundheit ist , kann auch ziemlich kompliziert sein .", "tgt": "and it can be a very complicated thing , what human health is ."}    
{"src": "nun , warum spielt das eine rolle für die menschliche gesundheit ?", "tgt": "now why does that matter for human health ?"}    
{"src": "das ist ein bild der cannery row von 1932 .", "tgt": "this is a shot of cannery row in 1932 ."}
```
### 处理脚本

目前的finetue数据主要是通过deltalm的提供的实现，通过脚本转换成封神数据格式

当前的转换脚本只是简单的将源语言和目标语言合并到一个文件，并生成上述格式，后续会继续完善处理脚本

脚本路径：Fengshenbang-LM/fengshen/examples/deltalm/prepare_dataset.py


使用方式：
```
python prepare_dataset.py processed_data_path de-en
```

## deltalm 模型

### deltalm模型路径
1) https://huggingface.co/IDEA-CCNL/Randeng-Deltalm-362M-En-Zn <br>
2) https://huggingface.co/IDEA-CCNL/Randeng-Deltalm-362M-Zh-En

主要包含三个文件：    
config.json：模型配置文件   
pytorch_model.bin：模型文件    
spm.model：sentence_piece文件    

### deltalm 模型结构
均实现在 Fengshenbang-LM/fengshen/models/deltalm 路径下，文件结构如下：    
1） modeling_deltalm.py 实现模型的基本结构，结构如论文所示    
2） tokenizer_deltalm.py 实现模型的tokenzier部分    
3） configuration_deltalm.py 实现模型的config配置部分    

### finetune 德译英示例
主要实现代码在 Fengshenbang-LM/fengshen/examples/translate/finetune_deltalm.py
通过脚本调用即可， 参考脚本 Fengshenbang-LM/fengshen/examples/translate/finetune_deltalm.sh

使用示例：
```
bash -x finetune_deltalm.sh 
```

注：如果要使用label_smoothing，当前需要设置label_smoothing参数不为0，当前默认值为0.1，直接修改finetune_deltalm.sh 对应参数值就可以

## 运行环境

pyhton = 3.8.10    
pytorch = 1.10.0    
transformers = 4.20.1    
pytorch-lightning = 1.6.5   

相关环境安装可参考Wiki：http://wiki.team.idea.edu.cn/pages/viewpage.action?pageId=16291924

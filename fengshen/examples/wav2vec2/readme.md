# Wav2vec2
本示例包括Wav2vec 2的预训练代码、finetune代码、对wenetspeech的预处理代码，在运行前请先安装ffmpeg（用于音频格式转换），并运行
```
pip install -r requirement.txt
```
安装需要的python包，如果kenlm无法安装，可以尝试运行
```pip install https://github.com/kpu/kenlm/archive/master.zip```
### 数据预处理
pretreatment中包括wenetspeech数据预处理的代码。如果要进行预训练，可以先下载wenetspeech数据集，解压并运行
  ```
  cd pretreatment
  ./pretrain_pretreatment.sh
  ```
如果要进行finetune，则需要进一步运行
  ```
  ./finetune_pretreatment.sh
  ```
来生成finetune所需的label
### 预训练
进行预训练需要修改pretrain_wav2vec2_base_wenet.sh，使得
* DATA_DIR为tsv文件所在的目录
* MODEL_PATH为config.json和preprocessor_config.json所在的目录
* HOME_PATH为保存模型checkpoint的目录

然后运行
  ```
  ./pretrain_wav2vec2_base_wenet.sh
  ```
### finetune
进行预训练需要修改finetune_ctc.sh，使得
* DATA_DIR为tsv文件所在的目录
* MODEL_PATH为config.json和preprocessor_config.json所在的目录
* PRETRAINED_PATH为预训练checkpoint所在的目录
* HOME_PATH为保存模型checkpoint的目录

然后运行
  ```
  ./finetune_ctc.sh
  ```
### ctc效果评估
运行ctc_metrics/ctc_metrics.sh会计算在dev、test_net、test_meeting三个测试集上的字错误率(cer)，并且生成包含模型预测结果的.tem的文件。运行前需要修改ctc_metrics.sh，使得
* DATA_HOME为dev、test_meeting、test_net三个数据集所在的目录
* CKPT为finetune的checkpoint所在的目录
* MODEL_PATH为config.json和vocab.json等配置文件所在的目录。

### 语言模型调分
如果要对ctc模型进行语言模型调分，需要准备kenlm语言模型。lm中提供了从wenetspeech中准备语料并训练语言模型的代码。首先需要先下载kenlm源码并编译
  ```
  git clone https://github.com/kpu/kenlm.git
  cd kenlm
  mkdir -p build
  cd build
  cmake ..
  make -j 4
  ```
编译完成后在build/bin目录下会生成训练和压缩模型需要的二进制文件。然后修改lm.sh中的参数使得
* exec_home为build/bin所在的绝对路径
* src为WenetSpeech.json的路径
* tgt_dir为保存语料的路径
* model_home为保存模型的路径

然后运行
  ```
  ./lm.sh
  ```
运行结束后会在model_home路径中生成.bin格式的模型。在ctc_metrics/inference.sh中指定lm_path为模型的路径即可使用语言模型进行调分。

### asr_demo
包含一个streamlit写的简单demo，能通过以下命令启动
  ```
  streamlit run demo.py
  ```
此demo可以对上传的音频文件进行语音识别。

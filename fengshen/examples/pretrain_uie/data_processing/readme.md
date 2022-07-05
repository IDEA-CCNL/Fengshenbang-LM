
# UIE_ZH 数据处理流程
UIE的数据流程稍微复杂，更多可以参考说明
- [https://github.com/universal-ie/UIE/tree/main/dataset_processing]

结构化数据
- 有监督数据【NER数据集【16】 RE数据集【5】 EE数据集【3】】
- 无监督数据 【Techkg 远监督得到的数据集】

中文NER/RE/EE 对应的数据的格式准备放置于：/cognitive_comp/yangjing/Fengshenbang-LM/dataset_processing/universal_ie/task_format，转换流程可以分为

- 第一步：对于不同任务准备好对应的数据格式 【 ChineseNER】【ChineseRE】 【ChineseEE】
- 第二步：配置datasets_name.yaml，在其中指定【mapper 标签映射】【path 数据路径】【split 数据划分】
- 第三步：使用 run_data_generation.sh 进行转换


原始数据和处理完成后的数据的链接，直接直接访问 [Google Cloud](https://drive.google.com/drive/folders/1mE20S3Yl0QxziYj_-eEiTmgB7wAITgy_?usp=sharing) 
- 原始数据: data_zh
- 处理完成后的数据: converted_data
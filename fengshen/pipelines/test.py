from fengshen.pipelines.text_classification import TextClassificationPipeline
import argparse
from datasets import load_dataset


# 预测 支持批量
pipe = TextClassificationPipeline(
    model='/data/gaoxinyu/pretrained_model/deberta-base-sp', device=-1)
print(pipe(['今天心情不好</s>今天很开心', '今天心情很好</s>今天很开心']))

# 训练 支持各种超参调整
total_parser = argparse.ArgumentParser("test")
total_parser = TextClassificationPipeline.add_data_specific_args(total_parser)
args = total_parser.parse_args()
datasets = load_dataset('IDEA-CCNL/AFQMC')
pipe = TextClassificationPipeline(
    args=args,
    model='/data/gaoxinyu/pretrained_model/deberta-base-sp', device=-1)
pipe.train(datasets)

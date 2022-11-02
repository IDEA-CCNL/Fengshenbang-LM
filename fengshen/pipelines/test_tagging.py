from fengshen.pipelines.sequence_tagging import SequenceTaggingPipeline
import argparse
import os

total_parser = argparse.ArgumentParser("test")
total_parser = SequenceTaggingPipeline.add_pipeline_specific_args(total_parser)
args = total_parser.parse_args()
args.data_dir="/cognitive_comp/lujunyu/data_zh/NER_Aligned/weibo"
args.gpus=2
args.max_epochs=30
args.decode_type='linear'
args.learning_rate=3e-5
args.strategy="deepspeed_stage_1"

os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
# pipe = SequenceTaggingPipeline(
#     model_path='/cognitive_comp/lujunyu/NER/outputs/ccks_crf/bert/best_checkpoint', args=args)
# print(pipe('李开复的哥哥在中国共产党读书。'))

pipe = SequenceTaggingPipeline(
    model_path='/cognitive_comp/lujunyu/XinYu/Fengshenbang-LM/fengshen/workspace/bert-base/pretrain', args=args)
pipe.train()





# Finetune Wenzhong Dialogue

## Usage

```bash
# finetune_wenzhong_dialogue.py    主文件
# finetune_wenzhong_dialogue.sh    是调试模式本地训练脚本
# cognitive_comp/yangqi/demo/wenzhong_demo.py demo 
# train
bash examples/wenzhong_dialo/finetune_wenzhong_dialogue.sh

# test during traiing

# test saved model
bash examples/wenzhong_dialo/finetune_wenzhong_dialogue.sh

```

## Update

- 1.0 完成 Wenzhong 模型的知识对话任务 Finetune（数据集 DuSinc）
- 1.1 完成 Eval 脚本（BLEU,DIST,F1）
- 1.2 新增 fs_dataset 多数据集 ConcatDataset 导入
- 1.3 新增 Mixing Sampler from T5 paper Sec. 3.2
- 1.4 新增全量多数据集 & 平衡多数据集两种方式代码脚本（主要是dataset subname 和 log 地址等修改）
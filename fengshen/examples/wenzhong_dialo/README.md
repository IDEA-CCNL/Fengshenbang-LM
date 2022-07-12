





## 运行代码 

```bash
# finetune_wenzhong_dialogue.py    主文件
# finetune_wenzhong_dialogue.sh    是调试模式本地训练脚本
# cognitive_comp/yangqi/demo/wenzhong_demo.py demo 
# train
bash examples/wenzhong_dialo/finetune_wenzhong_dialogue.sh

# cognitive_comp/yangqi/logs/wenzhong_dialo/Wenzhong-GOT-110M skpt & log 

# test
python examples/mt5_summary.py --gpus=1 --test_data=test_public.jsonl
--default_root_dir=/cognitive_comp/yangqi/logs/wenzhong_dialo/Wenzhong-GPT-110M
--do_eval_only
--resume_from_checkpoint=/cognitive_comp/yangqi/logs/wenzhong_dialo/Wenzhong-GPT-110M/ckpt/model-epoch=01-train_loss=1.9166.ckpt
--strategy=ddp

```


## Update
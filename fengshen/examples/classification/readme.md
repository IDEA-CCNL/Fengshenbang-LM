## 分类下游任务

在当前目录下，我们提供丰富的分类任务的示例，其中我们提供三个一键式运行的示例。

- demo_classification_afqmc_roberta.sh              使用DDP微调roberta
- demo_classification_afqmc_roberta_deepspeed.sh    结合deepspeed微调roberta，获得更快的运算速度
- demo_classification_afqmc_erlangshen_offload.sh   仅需7G显存即可微调我们效果最好的二郎神系列模型

上述示例均采用AFQMC的数据集，关于数据集的介绍可以在[这里](https://www.cluebenchmarks.com/introduce.html)找到。
同时我们处理过的数据文件已经放在Huggingface上，点击[这里](https://huggingface.co/datasets/IDEA-CCNL/AFQMC)直达源文件。
仅需要按我们的格式稍微处理一下数据集，即可适配下游不同的分类任务。
在脚本示例中，仅需要修改如下参数即可适配本地文件
```
        --dataset_name IDEA-CCNL/AFQMC \

-------> 修改为

        --data_dir $DATA_DIR \          # 数据目录
        --train_data train.json \       # 数据文件
        --valid_data dev.json \
        --test_data test.json \

```
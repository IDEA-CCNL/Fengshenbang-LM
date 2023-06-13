# 全参数Finetune
step1 将huggingface版本模型转化成fengshen的格式
```
sh convert_llama13b_to_fs.sh
```

step2 请按照自己机器的显存大小进行tensor并行切分，这里提供两个跑起来的示例供参考
1. 使用 3*8=24张 3090(24GB),Tensor parallesim = 8
2. 使用 1*8=8张 A100(80GB),Tensor parallesim = 1
```
sh convert_llama13b_tp8.sh
```

step3 训练

N=1/2, TP=4/8 OOM
N=3, TP=8 跑起来,

# Lora微调
TODO



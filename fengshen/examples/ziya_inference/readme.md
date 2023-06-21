# fengshen-infer

# fengshen-infer

### 量化推理

量化分为3阶段:
- 开源量化方案
- 开源量化方案+模型并行
- 开源量化方案+模型并行+模型结构优化

#### 第一阶段：accelerate

**a).为方便大众使用，需要安装的库和依赖较少，容易上手。主要是hugging face社区提供的量化推理加速方案。**

代码见：[hf_quantizatin_inference.py](http://git.team.idea.edu.cn/cognitive-computing/fengshen-infer/-/tree/dev_wzw)

以开源版本模型Ziya-13B为例，性能如下：

|机器配置|精度|显存占用|batch_size|per_token_time_cost|c_eval得分|
|:----:|:--:|:-----:|:--------:|:-----------------:|:--------:|
|2X3090|fp16|26GB|1|49.89 ms|32.29|
|4X3090|fp16|26GB|1|28.65 ms|32.29|
|1X3090|int8|13GB|1|147.06 ms|31.29|
|1X3090|int4|-|1|-|-|

 accelerate的int4方案在transformer最新分支有，安装版本还没有。

**b).基于[llama.cpp](https://github.com/ggerganov/llama.cpp)方案优化

llama.cpp能实现在MacOS等边缘平台部署llama模型，并使用cpu进行推理。
CPU Benchmark见：
[cpu_benchmark](https://github.com/ggerganov/llama.cpp#quantization)


代码见：[llama_cpp_quantizatin_inference.py](http://git.team.idea.edu.cn/cognitive-computing/fengshen-infer/-/tree/dev_zyh)

开源版本模型Ziya-13B为例，性能如下：

|机器配置|精度|显存占用|batch_size|per_token_time_cost|c_eval得分|
|:----:|:--:|:-----:|:--------:|:-----------------:|:--------:|
|2X3090|fp16|26GB|1|53.29 ms|-|
|1X3090|q8_0|13GB|1|46.25 ms|29.47|
|1X3090|q5_0|8.4GB|1|60.37 ms|29.65|
|1X3090|q5_1|9.2GB|1|59.84 ms|30.46|
|1X3090|q4_0|6.9GB|1|45.05 ms|25.53|
|1X3090|q4_1|7.7GB|1|46.87 ms|30.89|
- q4_0 = 32 numbers in chunk, 4 bits per weight, 1 scale value at 32-bit float (5 bits per value in average), each weight is given by the common scale * quantized value.

- q4_1 = 32 numbers in chunk, 4 bits per weight, 1 scale value and 1 bias value at 32-bit float (6 bits per value in average), each weight is given by the common scale * quantized value + common bias.

- q5_0 = 32 numbers in chunk, 5 bits per weight, 1 scale value at 16-bit float, size is 5.5 bits per weight.

- q5_1 = 32 numbers in a chunk, 5 bits per weight, 1 scale value at 16 bit float and 1 bias value at 16 bit, size is 6 bits per weight.

- q8_0 = same as q4_0, except 8 bits per weight, 1 scale value at 32 bits, making total of 9 bits per weight.


#### 第二阶段
>进行中...

### 参考资料
- accelerate
- llama.cpp-python
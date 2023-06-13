"""
dependencies
    llama.cpp (https://github.com/ggerganov/llama.cpp)
    llama-cpp-python (https://github.com/abetlen/llama-cpp-python)

    llama.cpp
        1. 通过llama.cpp将模型转换为ggml格式
        2. 参考llama.cpp对转换后的模型量化到 (q4_0, q4_1, q5_0, q5_1, q8_0)
            - 在转换过程中会遇到tokenizer对不齐的问题，自行在词表中添加相应个数token即可
        3. 依据自身环境执行MAKE或CMAKE命令 (若使用GPU则应有相应的cuda-toolkit)
        4. ./main -m $(model_path) 运行
    llama-cpp-python
        1. 参考 https://abetlen.github.io/llama-cpp-python/#llama_cpp.Llama
        2. 若要使用gpu, 需要在conda环境内安装合适的cuda-toolkit
        3. 执行 CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python 安装命令使用GPU
"""


from llama_cpp import Llama
import numpy as np

"""
以下提供一个基于llama-cpp-python较为简单的脚本
即llama-cpp-python中的high-level-api
以及
"""

# load llama.cpp ggml format model
# MODEL_PATH = "/models/ggml/ggml-model-q5_0.bin"
MODEL_PATH = "/data0/zhangyuhan/llama_cpp/ggml-model-q5_0.bin"
llm = Llama(model_path=MODEL_PATH, verbose=True, n_gpu_layers=40)

#infer
output = llm("<human>: 林黛玉葬花这一情节出自哪个国家的著作？ <bot>: ", max_tokens=32, stop=["<human>:"], echo=True)
print(output)

#generator
def stop_criteria(inputid, logits):
    #直接设定为ziya结束的tokenid: 2
    return np.argmax(logits) == 2

query = "<human>: 林黛玉葬花这一情节出自哪个国家的著作？ <bot>: ".encode("utf-8")
tokens = llm.tokenize(text = bytes(query))
for token in llm.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1, stopping_criteria=stop_criteria):
    print(llm.detokenize([token]).decode("utf-8"))


# 获取chimed的数据，并选取打分最高的topk(default k=10w)数据
import json
input_path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/chimed.trainrm_scored.json'
output_path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/chimed.trainrm_scored_top10w.json'

import json
import heapq

def get_top_rewards(json_file_path, output_file_path, top_n=100000):
    # 建立一个空的最小堆
    min_heap = []

    with open(json_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            reward = data['reward'][0]  # 获取reward值
            # 当堆中的数据小于10w条时，直接添加到堆中
            if len(min_heap) < top_n:
                heapq.heappush(min_heap, (reward, line))
            # 否则，只有当前数据的reward大于堆顶的reward时才替换堆顶的数据
            else:
                if reward > min_heap[0][0]:
                    heapq.heappushpop(min_heap, (reward, line))

    # 将堆中的数据写入到输出文件中
    with open(output_file_path, 'w') as f:
        for _, line in min_heap:
            f.write(line)

if __name__ == "__main__":
    get_top_rewards(input_path, output_path)


import argparse
from fengshen import UbertPiplines
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser = UbertPiplines.piplines_args(total_parser)
    args = total_parser.parse_args()

    # 设置一些训练要使用到的参数
    args.default_root_dir = './'
    args.max_epochs = 5
    args.gpus = 1
    args.batch_size = 1

    # 只需要将数据处理成为下面数据的 json 样式就可以一键训练和预测，下面只是提供了一条示例样本
    train_data = [
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，",
            "choices": [
                {"entity_type": "地址", "label": 0, "entity_list": [
                    {"entity_name": "台湾", "entity_type": "地址", "entity_idx": [[15, 16]]}]},
                {"entity_type": "书名", "label": 0, "entity_list": []},
                {"entity_type": "公司", "label": 0, "entity_list": []},
                {"entity_type": "游戏", "label": 0, "entity_list": []},
                {"entity_type": "政府机构", "label": 0, "entity_list": []},
                {"entity_type": "电影名称", "label": 0, "entity_list": []},
                {"entity_type": "人物姓名", "label": 0, "entity_list": [
                    {"entity_name": "彭小军", "entity_type": "人物姓名", "entity_idx": [[0, 2]]}]},
                {"entity_type": "组织机构", "label": 0, "entity_list": []},
                {"entity_type": "岗位职位", "label": 0, "entity_list": []},
                {"entity_type": "旅游景点", "label": 0, "entity_list": []}
            ],
            "id": 0}
    ]
    dev_data = [
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "就天涯网推出彩票服务频道是否是业内人士所谓的打政策“擦边球”，记者近日对此事求证彩票监管部门。",
            "choices": [
                {"entity_type": "地址", "label": 0, "entity_list": []},
                {"entity_type": "书名", "label": 0, "entity_list": []},
                {"entity_type": "公司", "label": 0, "entity_list": [
                    {"entity_name": "天涯网", "entity_type": "公司", "entity_idx": [[1, 3]]}]},
                {"entity_type": "游戏", "label": 0, "entity_list": []},
                {"entity_type": "政府机构", "label": 0, "entity_list": []},
                {"entity_type": "电影名称", "label": 0, "entity_list": []},
                {"entity_type": "人物姓名", "label": 0, "entity_list": []},
                {"entity_type": "组织机构", "label": 0, "entity_list": [
                    {"entity_name": "彩票监管部门", "entity_type": "组织机构", "entity_idx": [[40, 45]]}]},
                {"entity_type": "岗位职位", "label": 0, "entity_list": [
                    {"entity_name": "记者", "entity_type": "岗位职位", "entity_idx": [[31, 32]]}]},
                {"entity_type": "旅游景点", "label": 0, "entity_list": []}
            ],

            "id": 0}

    ]
    test_data = [
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。",
            "choices": [
                {"entity_type": "地址", "label": 0, "entity_list": [
                    {"entity_name": "雅清苑", "entity_type": "地址", "entity_idx": [[12, 14]]}]},
                {"entity_type": "书名", "label": 0, "entity_list": []},
                {"entity_type": "公司", "label": 0, "entity_list": []},
                {"entity_type": "游戏", "label": 0, "entity_list": []},
                {"entity_type": "政府机构", "label": 0, "entity_list": []},
                {"entity_type": "电影名称", "label": 0, "entity_list": []},
                {"entity_type": "人物姓名", "label": 0, "entity_list": []},
                {"entity_type": "组织机构", "label": 0, "entity_list": []},
                {"entity_type": "岗位职位", "label": 0, "entity_list": [
                    {"entity_name": "公务员", "entity_type": "岗位职位", "entity_idx": [[18, 20]]}]},
                {"entity_type": "旅游景点", "label": 0, "entity_list": []}
            ],
            "id": 0},
    ]

    model = UbertPiplines(args)
    model.fit(train_data, dev_data)
    result = model.predict(test_data)
    for line in result:
        print(line)


if __name__ == "__main__":
    main()

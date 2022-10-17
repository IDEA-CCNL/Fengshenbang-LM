import argparse
from fengshen.pipelines.multiplechoice import UniMCPiplines


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser = UniMCPiplines.piplines_args(total_parser)
    args = total_parser.parse_args()

    pretrained_model_path = 'IDEA-CCNL/Erlangshen-UniMC-RoBERTa-110M-Chinese'
    args.learning_rate = 2e-5
    args.max_length = 512
    args.max_epochs = 3
    args.batchsize = 8
    args.train = 'train'
    args.default_root_dir = './'

    model = UniMCPiplines(args, model_path=pretrained_model_path)

    train_data = [    # 训练数据
        {
            "texta": "凌云研发的国产两轮电动车怎么样，有什么惊喜？",
            "textb": "",
            "question": "下面新闻属于哪一个类别？",
            "choice": [
                "教育",
                "科技",
                "军事",
                "旅游",
                "国际",
                "股票",
                "农业",
                "电竞"
            ],
            "answer": "科技",
            "label": 1,
            "id": 0
        }
    ]
    dev_data = [     # 验证数据
        {
            "texta": "我四千一个月，老婆一千五一个月，存款八万且有两小孩，是先买房还是先买车？",
            "textb": "",
            "question": "下面新闻属于哪一个类别？",
            "choice": [
                "故事",
                "文化",
                "娱乐",
                "体育",
                "财经",
                "房产",
                "汽车"
            ],
            "answer": "汽车",
            "label": 6,
            "id": 0
        }
    ]
    test_data = [    # 测试数据
        {"texta": "街头偶遇2018款长安CS35，颜值美炸！或售6万起，还买宝骏510？",
         "textb": "",
         "question": "下面新闻属于哪一个类别？",
         "choice": [
             "房产",
             "汽车",
             "教育",
             "军事"
         ],
         "answer": "汽车",
         "label": 1,
         "id": 7759}
    ]

    if args.train:
        model.train(train_data, dev_data)
    result = model.predict(test_data)
    for line in result:
        print(line)


if __name__ == "__main__":
    main()

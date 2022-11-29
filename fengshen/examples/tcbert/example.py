import argparse
from fengshen.pipelines.tcbert import TCBertPipelines

def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser = TCBertPipelines.piplines_args(total_parser)
    args = total_parser.parse_args()

    pretrained_model_path = 'IDEA-CCNL/Erlangshen-UniMC-RoBERTa-110M-Chinese'
    args.learning_rate = 2e-5
    args.max_length = 512
    args.max_epochs = 3
    args.batchsize = 8
    args.train = 'train'
    args.default_root_dir = './'

    train_data = [    # 训练数据
        {
            "content": "凌云研发的国产两轮电动车怎么样，有什么惊喜？",
            "label": "科技",
        }
    ]
    dev_data = [     # 验证数据
        {
            "content": "我四千一个月，老婆一千五一个月，存款八万且有两小孩，是先买房还是先买车？",
            "label": "汽车",
        }
    ]
    test_data = [    # 测试数据
        {"content": "街头偶遇2018款长安CS35，颜值美炸！或售6万起，还买宝骏510？",
        }
    ]

    prompt_label = {"故事": "故事", "文化": "文化",    #标签映射
                    "娱乐": "娱乐", "体育": "体育", 
                    "财经": "财经", "房产": "房产", 
                    "汽车": "汽车", "教育": "教育", 
                    "科技": "科技", "军事": "军事", 
                    "旅游": "旅游", "国际": "国际", 
                    "股票": "股票", "农业": "农业", 
                    "游戏": "游戏"}

    model = TCBertPipelines(args, model_path=pretrained_model_path, nlabels=len(prompt_label))

    if args.train:
        model.train(train_data, dev_data, prompt_label)
    result = model.predict(test_data, prompt_label)
    for line in result:
        print(line)


if __name__ == "__main__":
    main()

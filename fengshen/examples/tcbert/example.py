import argparse
from fengshen.pipelines.tcbert import TCBertPipelines
from pytorch_lightning import seed_everything

def main():
    seed_everything(123)
    total_parser = argparse.ArgumentParser("Topic Classification")
    total_parser = TCBertPipelines.piplines_args(total_parser)
    args = total_parser.parse_args()

    pretrained_model_path = 'IDEA-CCNL/Erlangshen-TCBert-110M-Classification-Chinese'
    args.learning_rate = 2e-5
    args.max_length = 512
    args.max_epochs = 5
    args.batchsize = 4
    args.train = 'train'
    args.default_root_dir = './'
    # args.gpus = 1   #注意：目前使用CPU进行训练，取消注释会使用GPU，但需要配置相应GPU环境版本
    args.fixed_lablen = 2 #注意：可以设置固定标签长度，由于样本对应的标签长度可能不一致，建议选择适中的数值表示标签长度

    train_data = [    # 训练数据
        {"content": "真正的放养教育，放的是孩子的思维，养的是孩子的习惯", "label": "故事"},
        {"content": "《唐人街探案》捧红了王宝强跟刘昊然，唯独戏份不少的他发展最差", "label": "娱乐"},
        {"content": "油价攀升 阿曼经济加速增长", "label": "财经"},
        {"content": "日本男篮近期动作频频，中国队的未来劲敌会是他们吗？", "label": "体育"},
        {"content": "教育部：坚决防止因撤并乡村小规模学校导致学生上学困难", "label": "教育"},
        {"content": "LOL设计最完美的三个英雄，玩家们都很认可！", "label": "电竞"},
        {"content": "上联：浅看红楼终是梦，怎么对下联？", "label": "文化"},
        {"content": "楼市再出新政！北京部分限房价项目或转为共有产权房", "label": "房产"},
        {"content": "企业怎样选云服务器？云服务器哪家比较好？", "label": "科技"},
        {"content": "贝纳利的三缸车TRE899K、TRE1130K华丽转身", "label": "汽车"},
        {"content": "如何评价：刘姝威的《严惩做空中国股市者》？", "label": "股票"},
        {"content": "宁夏邀深圳市民共赴“寻找穿越”之旅", "label": "旅游"},
        {"content": "日本自民党又一派系力挺安倍 称会竭尽全力", "label": "国际"},
        {"content": "农村养老保险每年交5000，交满15年退休后能每月领多少钱？", "label": "农业"},
        {"content": "国产舰载机首次现身，进度超过预期，将率先在滑跃航母测试", "label": "军事"}
    ]

    dev_data = [     # 验证数据
        {"content": "西游记后传中，灵儿最爱的女人是谁？不是碧游！", "label": "故事"},
        {"content": "小李子莱奥纳多有特别的提袋子技能，这些年他还有过哪些神奇的造型？", "label": "娱乐"},
        {"content": "现在手上有钱是投资买房还是存钱，为什么？", "label": "财经"},
        {"content": "迪卡侬的衣服值得购买吗？", "label": "体育"},
        {"content": "黑龙江省旅游委在齐齐哈尔组织举办导游培训班", "label": "教育"},
        {"content": "《王者荣耀》中，哪些英雄的大招最“废柴”？", "label": "电竞"},
        {"content": "上交演绎马勒《复活》，用音乐带来抚慰和希望", "label": "文化"},
        {"content": "All in服务业，58集团在租房、住房市场的全力以赋", "label": "房产"},
        {"content": "为什么有的人宁愿选择骁龙660的X21，也不买骁龙845的小米MIX2S？", "label": "科技"},
        {"content": "众泰大型SUV来袭，售13.98万，2.0T榨出231马力，汉兰达要危险了", "label": "汽车"},
        {"content": "股票放量下趺，大资金出逃谁在接盘？", "label": "股票"},
        {"content": "广西博白最大的特色是什么？", "label": "旅游"},
        {"content": "特朗普退出《伊朗核协议》，对此你怎么看？", "label": "国际"},
        {"content": "卖水果利润怎么样？", "label": "农业"},
        {"content": "特种兵都是身材高大的猛男么？别再被电视骗了，超过1米8都不合格", "label": "军事"}
    ]

    test_data = [    # 测试数据
        {"content": "廖凡重出“江湖”再争影帝 亮相戛纳红毯霸气有型"},
        {"content": "《绝地求生: 刺激战场》越玩越卡？竟是手机厂商没交“保护费”!"},
        {"content": "买涡轮增压还是自然吸气车？今天终于有答案了！"},
    ]

    #标签映射  将真实标签可以映射为更合适prompt的标签 
    prompt_label = {  
                    "体育":"体育", "军事":"军事", "农业":"农业",  "国际":"国际", 
                    "娱乐":"娱乐", "房产":"房产", "故事":"故事",  "教育":"教育",
                    "文化":"文化", "旅游":"旅游", "汽车":"汽车",  "电竞":"电竞", 
                    "科技":"科技", "股票":"股票", "财经":"财经"
                    }
    
    #不同的prompt会影响模型效果
    #prompt = "这一句描述{}的内容如下："
    prompt = "下面是一则关于{}的新闻："
                    
    model = TCBertPipelines(args, model_path=pretrained_model_path, nlabels=len(prompt_label))

    if args.train:
        model.train(train_data, dev_data, prompt, prompt_label)
    result = model.predict(test_data, prompt, prompt_label)

    for i, line in enumerate(result):
        print({"content":test_data[i]["content"], "label":list(prompt_label.keys())[line]})


if __name__ == "__main__":
    main()

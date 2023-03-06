example_dict={
    "文本分类":{"text_a":"钢琴块3别踩白块儿3钢琴块3是一款简洁的钢琴模拟软件,在Android平台上,类似的软件还是比较多的。","choices":["相机","影视娱乐","棋牌中心","新闻","财经","策略","休闲益智","教育"]},
    '新闻分类':{"text_a":"微软披露拓扑量子计算机计划！","choices":["故事","文化","娱乐","体育","财经","房产","汽车","教育","科技"]},
    '情感分析':{"text_a":"刚买iphone13 pro 还不到一个月，天天死机最差的一次购物体验","choices":["好评","差评"]},
    '意图识别':{"text_a":"打电话给吴小军。","choices":["放音乐","播放下一首","打电话","退出导航","开始导航","其他","暂停音乐","导航","开导航"]},

    '语义匹配':{"text_a":"今天心情不好","text_b":"我很不开心","choices":["相似","不相似"]},
    '自然语言推理':{"text_a":"小明正在上高中","text_b":"小明是一个初中生","choices":["无关","矛盾","蕴含"]},

    '多项选择':{"text_a":"这大家千万不能着急，我们现在只是暂时输了7分。距离比赛结束还有20多分钟呢，我们是完全有机会转败为赢的，大家加油!","question":"说话人希望大家：","choices":["别得意","冷静一些","加快速度","提前预习"]},
    '指代消解':{"text_a":"李鸣觉得董客这人，踏实得叫人难受。可因为孟野和森森太疯，他只好去找董客聊天，但在董客眼里，李鸣也是不正常，他竟然放着现成的大学不愿上。","question":"【他】指的是【李鸣】吗？","choices":["是","不是"]},

    '实体识别':{"text_a":"北京大学是我国的一座历史名校，坐落在海淀区，蔡元培曾经担任校长","question":"机构"},
    '抽取式阅读理解':{"text_a":"《H》正式定档3月7日下午两点整在京东商城独家平台开启第一批5000份预售,定价230元人民币,回馈最忠实的火星歌迷,意在用精品回馈三年来跟随华晨宇音乐不离不弃的粉丝们的支持与厚爱","question":"华晨宇专辑h预售价格是多少？"},
    '关键词抽取':{"text_a":"今儿在大众点评，找到了口碑不错的老茶故事私房菜。"},

    "生成式摘要":{"text_a":"针对传统的流量分类管理系统存在不稳定、结果反馈不及时、分类结果显示不直观等问题,设计一个基于web的在线的流量分类管理系统.该系统采用流中前5个包(排除3次握手包)所含信息作为特征值计算资源,集成一种或多种分类算法用于在线网络流量分类,应用数据可视化技术处理分类结果.实验表明:在采用适应在线分类的特征集和c4.5决策树算法做分类时,系统能快速做出分类,且精度达到94％以上;数据可视化有助于人机交互,改善分类指导."}
}


# 构造prompt的过程中，verbalizer这个占位key的内容，是通过 "/".join(choices) 拼接起来
dataset2instruction = {
    "情感分析": {
        "prompt": "{}任务：【{}】这篇文章的情感态度是什么？{}",
        "keys_order": ["subtask_type","text_a", "verbalizer"],
        "data_type": "classification",
    },
    "文本分类": {
        "prompt": "{}任务：【{}】这篇文章的类别是什么？{}",
        "keys_order": ["subtask_type","text_a", "verbalizer"],
        "data_type": "classification",
    },
    "新闻分类": {
        "prompt": "{}任务：【{}】这篇文章的类别是什么？{}",
        "keys_order": ["subtask_type","text_a", "verbalizer"],
        "data_type": "classification",
    },
    "意图识别": {
        "prompt": "{}任务：【{}】这句话的意图是什么？{}",
        "keys_order": ["subtask_type","text_a", "verbalizer"],
        "data_type": "classification",
    },
# --------------------
    "自然语言推理": {
        "prompt": "{}任务：【{}】和【{}】，以上两句话的逻辑关系是什么？{}",
        "keys_order": ["subtask_type","text_a", "text_b", "verbalizer"],
        "data_type": "classification",
    },
    "语义匹配": {
        "prompt": "{}任务：【{}】和【{}】，以上两句话的内容是否相似？{}",
        "keys_order": ["subtask_type","text_a", "text_b", "verbalizer"],
        "data_type": "classification",
    },
# -----------------------
    "指代消解": {
        "prompt": "{}任务：文章【{}】中{}{}",
        "keys_order": ["subtask_type","text_a", "question", "verbalizer"],
        "data_type": "classification",
    },
    "多项选择": {
        "prompt": "{}任务：阅读文章【{}】问题【{}】？{}",
        "keys_order": ["subtask_type","text_a", "question", "verbalizer"],
        "data_type": "classification",
    },
# ------------------------
    "抽取式阅读理解": {
        "prompt": "{}任务：阅读文章【{}】问题【{}】的答案是什么？",
        "keys_order": ["subtask_type","text_a", "question"],
        "data_type": "mrc",
    },
    "实体识别": {
        "prompt": "{}任务：找出【{}】这篇文章中所有【{}】类型的实体？",
        "keys_order": ["subtask_type","text_a", "question"],
        "data_type": "ner",
    },
# ------------------------
    "关键词抽取": {
        "prompt": "{}任务：【{}】这篇文章的关键词是什么？",
        "keys_order": ["subtask_type","text_a"],
        "data_type": "keys",
    },
    "关键词识别":{
        "prompt": "{}任务：阅读文章【{}】问题【{}】{}",
        "keys_order": ["subtask_type","text_a","question","verbalizer"],
        "data_type": "classification",
    },
    "生成式摘要": {
        "prompt": "{}任务：【{}】这篇文章的摘要是什么？",
        "keys_order": ["subtask_type","text_a"],
        "data_type": "summ",
    },
}

def get_instruction(sample):

    template = dataset2instruction[sample["subtask_type"]]
    # print(template)
    # print(sample)
    sample["instruction"] = template["prompt"].format(*[
                sample[k] for k in template["keys_order"]
            ])

    print(sample["instruction"])

    return sample["instruction"]

sample1 = {"subtask_type": "生成式摘要",
        "text_a":"中彬县非企业委员会文件中国网络资讯台记者获悉，7月23日，中共彬县非公有制企业委员会发红头文件组织给7月19日发生火灾的家福乐超市捐款，火灾给家福乐超市造成较大损失，该文件要求各支部，各企业要迅速安排，组织捐款。此事令彬县群众震惊，群众给县长写“太不该”打油诗。<Paragraph>群众给县长写“太不该”打油诗。中国网络资讯台记者获悉网民热议彬县非公企业党委倡导县内部分非公企业向家福乐购物有限公司捐款一事后，县委、县政府高度重视，立即组织调查组进行调查。经初步调查核实，县非公企业党委在未向县委、县政府报告同意的情况下，向县内部分非公企业下发了《关于向彬县家福乐购物有限公司捐款的通知》。县委、县政府认为，在家福乐购物中心二店火灾事故原因正在调查期间，县非公企业党委组织这样的捐款活动是不妥当的。因此，县委、县政府已责成县非公企业党委收回该文件，停止捐款活动。关于该起火灾事故原因，省、市消防部门正在做深入调查。"
         }
text2 = get_instruction(sample1)
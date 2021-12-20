#!/usr/bin/env python

from typing import Literal

from torch.utils.data import Dataset, random_split


def train_test_split(dataset: Dataset, test_size=0.25):
    if 0 < test_size < 1:
        test_size = int(test_size * len(dataset))
    train_size = len(dataset) - test_size

    return random_split(dataset, [train_size, test_size])


FEATURE_SIZE = {
    "grid": (2048, 7, 7),
    "roi": (256, 7, 7),
    "e2e": (2048, 7, 7),
}

ALI_CATE_LEVEL_NAME = Literal[
    "女装_女士精品", "女鞋", "男装", "服饰配件_皮带_帽子_围巾", "流行男鞋", "运动服_休闲服装", "运动鞋new"
]

ALI_CATE_NAME = Literal[
    "连衣裙",
    "T恤",
    "时装凉鞋",
    "单鞋",
    "背心_马甲",
    "马夹",
    "风衣",
    "短外套",
    "大码女装",
    "一字拖",
    "包头拖",
    "中老年女装",
    "蕾丝衫_雪纺衫",
    "休闲裤",
    "手套",
    "时尚套装",
    "休闲运动套装",
    "衬衫",
    "棉衣_棉服",
    "帆布鞋",
    "雨鞋",
    "工装制服",
    "鞋垫",
    "夹克",
    "礼服_晚装",
    "婚纱",
    "工装鞋",
    "牛仔裤",
    "上衣",
    "毛针织衫",
    "帽子",
    "其他配件",
    "POLO衫",
    "卫衣_绒衫",
    "休闲皮鞋",
    "乐福鞋（豆豆鞋）",
    "钥匙扣",
    "酒店工作制服",
    "运动连衣裙",
    "针织衫_毛衣",
    "西裤",
    "民族服装_舞台装",
    "连体衣_裤",
    "半身裙",
    "卫衣",
    "毛呢外套",
    "毛衣",
    "西装",
    "羽绒服",
    "时装靴",
    "运动背心",
    "运动内衣套装",
    "前掌垫",
    "板鞋_休闲鞋",
    "时尚休闲鞋",
    "网面鞋",
    "棉衣",
    "学生校服",
    "布鞋",
    "高帮鞋",
    "打底裤",
    "运动内裤",
    "运动套装",
    "毛线",
    "抹胸",
    "休闲板鞋",
    "运动T恤",
    "健身衣",
    "跑步鞋",
    "运动棉衣",
    "其他套装",
    "围巾_丝巾_披肩",
    "汉服",
    "背心吊带",
    "运动卫衣_套头衫",
    "三件套",
    "篮球鞋",
    "西服套装",
    "毛呢大衣",
    "西服",
    "运动风衣",
    "皮草",
    "增高垫",
    "马丁靴",
    "其它制服_套装",
    "运动中长裤／短裤",
    "人字拖",
    "运动茄克_外套",
    "腰带_皮带_腰链",
    "包挂件",
    "裙子",
    "运动长裤",
    "正装皮鞋",
    "皮衣",
    "职业女裤套装",
    "时尚休闲沙滩鞋",
    "拎环",
    "包带",
    "耳套",
    "运动半身裙",
    "足球鞋",
    "皇冠配饰",
    "大码上装",
    "裤子",
    "跑步T恤",
    "运动牛仔裤",
    "领结",
    "领带",
    "沙滩鞋",
    "时尚雪地靴",
    "职业女裙套装",
    "切尔西靴",
    "智能手环",
    "旗袍",
    "洞洞鞋",
    "运动羽绒服",
    "民族服装",
    "运动沙滩鞋_凉鞋",
    "健身裤",
    "防晒面纱",
    "新娘头饰",
    "袖扣",
    "Polo衫",
    "弹力靴_袜靴",
    "运动文胸",
    "其他拖鞋",
    "其他民族服装",
    "棒球服",
    "健身套装",
    "多件套",
    "布面料",
    "松糕（摇摇）鞋",
    "单马甲",
    "鞋带",
    "唐装",
    "穆勒鞋",
    "防晒袖套",
    "跑步裤",
    "中老年上装",
    "头纱",
    "中老年套装",
    "童鞋_青少年鞋",
    "中山装",
    "运动毛衣_线衫",
    "马甲",
    "大码下装",
    "西装裤_正装裤",
    "口袋巾",
    "运动拖鞋",
    "运动POLO衫",
    "羽毛球鞋",
    "假领",
    "防滑贴",
    "其它运动鞋",
    "羊绒开衫",
    "综合训练鞋_室内健身鞋",
    "医护制服",
    "指环",
    "卫裤",
    "鞋扣",
    "领带夹",
    "皮裤",
    "休闲西服",
    "皮带扣",
    "棉裤",
    "领针",
    "手帕",
    "中老年下装",
    "棉裤_羽绒裤",
    "鞋撑",
    "其他凉鞋",
    "婚纱裙撑",
    "保健环",
    "后跟贴",
    "乒乓球鞋",
    "羽绒裤",
    "跑步外套",
    "二件套",
    "健步鞋",
    "网球鞋",
    "婚纱手套",
    "羽绒马甲",
]

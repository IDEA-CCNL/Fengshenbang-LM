import json
import jieba
import os
import argparse
import re


def num_to_ch(num):
    """
    功能说明：将阿拉伯数字 ===> 转换成中文数字（适用于[0, 10000)之间的阿拉伯数字 ）
    """
    try:
        num = int(num)
    except Exception:
        return ""
    _MAPPING = ('零', '一', '二', '三', '四', '五', '六', '七', '八', '九', )
    _P0 = ('', '十', '百', '千', )
    _S4 = 10 ** 4
    if num < 0 or num >= _S4:
        return None
    if num < 10:
        return _MAPPING[num]
    else:
        lst = []
        while num >= 10:
            lst.append(num % 10)
            num = num // 10
        lst.append(num)
        c = len(lst)    # 位数
        result = ""
        for idx, val in enumerate(lst):
            if val != 0:
                result += _P0[idx] + _MAPPING[val]
            if idx < c - 1 and lst[idx + 1] == 0:
                result += u'零'
        result = result[::-1]
        if result[:2] == "一十":
            result = result[1:]
        if result[-1:] == "零":
            result = result[:-1]
        return result


def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff' or ch == '\ufeff':
            return True
    return False


def is_ustr(in_str):  # 去除非中文和数字的
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str+in_str[i]
        else:
            out_str = out_str
    return out_str


def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return False
    if uchar in ('-', ',', '，', '。', '.', '>', '?'):
        return False
    return False


def clean(x):  # 两个字符间插入空白字符
    # step-1: 去除英文和标点符号
    r1 = '[a-zA-Z’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~：| ☆；．（）—～]+'
    x = re.sub(r1, '', x)
    # step-2: 去除非中文和数字的
    x = is_ustr(x)
    # step-2: 分词并且将数字改成大写
    # step-3: 将句子用空格拼接起来
    try:
        y = [num_to_ch(a) if not is_Chinese(a) else a for a in jieba.lcut(x)]
        return " ".join(y)
    except Exception:
        print(jieba.lcut(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src")
    parser.add_argument("--tgt_dir")
    args = parser.parse_args()
    with open(args.src) as f:
        data = json.load(f)
    audios = data["audios"]
    text = []
    for audio in audios:
        if "train" not in audio["path"]:
            continue
        if "segments" in audio:
            for segment in audio["segments"]:
                if "text" in segment:
                    text.append(clean(segment["text"]))

    os.makedirs(args.tgt_dir, exist_ok=True)
    with open(os.path.join(args.tgt_dir, "lm_data.txt"), "w") as f:
        f.write("\n".join(text))

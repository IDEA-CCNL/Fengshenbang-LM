# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : test.py
#   Last Modified : 2022-07-18 10:44
#   Describe      : 
#
# ====================================================
import re


ANS_RE = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)$")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            match_str = float(match_str)
            match_str = round(match_str, 3)
            match_str = str(match_str)
        except:
            print("matched but not a float", match_str)
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

if __name__ == "__main__":
    a = "11 - 5 = <<11-5=6>>6 players were cautioned. \
    They would receive 6 of the 11 soccer players received a caution: 5 + 1 = <<5+1=6>>6 * 1 =6>>6 yellow cards. \
    Knowing that each red card corresponds to 2 yellow cards,6>>6 \
    Knowing that each red card corresponds to 2 yellow cards, the team would have a total of 6 / 2 = <<6/2=3>>3 red cards. \
    [ANS] 3 * 6 = <<6*2=12>>12 red cards. \
    [ANS] 12"
    res = extract_answer(a)
    print(res)


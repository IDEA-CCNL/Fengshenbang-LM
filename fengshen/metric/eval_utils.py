import sys, re
sys.path.append("../")
from fengshen.utils import chinese_char_tokenize #内部分词

from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import jieba
from collections import Counter

# utils
def load_metric_fn(metric):
    fn_dict = {
        "bleu": bleu_fn,
        "f1" : f1_fn,
        "rouge": rouge_fn,
        "em" : em_fn,
        "acc": acc_fn,
        "dist": dist_fn,
        "vocab": vocab_fn
    }
    if metric in fn_dict.keys():
        return fn_dict[metric]
    else:
        raise NotImplementedError

def mean(x):
    return sum(x)/len(x)

def normalize(x):
    return x 

# ! deprecated now
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.\n
    The four following functions are adapted from Meta ParlAI:https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/metrics.py
    """
    re_art = re.compile(r"\b(是|的|啊)\b")
    re_punc = re.compile(r"[!\"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\'，。？！]")
    def remove_articles(text):
        return re_art.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return re_punc.sub(" ", text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def bleu_fn(references, candidates):
    references = [chinese_char_tokenize(ref[0]) for ref in references]
    candidates = [chinese_char_tokenize(can) for can in candidates]
    scores_list = {}
    for i in range(1,5):
        scores_list['bleu_'+str(i)] = bleu_score(candidates,references,i,False)

    return scores_list

def ntlk_bleu(references, candidates):
    for ref, can in zip(references, candidates):
        reference = [[ref]]
        candidate = [can]

        chencherry = SmoothingFunction()
        score1, score2= sentence_bleu(reference,candidate,weights=[(1.0,),(0.5,0.5)]) #methods 7 ref prophetnet

    return score1,score2
    
def rouge_fn(references, candidates):
    references = [chinese_char_tokenize(ref[0]) for ref in references]
    candidates = [chinese_char_tokenize(can) for can in candidates]

    scores_list = rouge_score(candidates,references,normalizer=normalize)
    return scores_list

def em_fn(references, candidates):
    def em(ref,can):
        common = Counter(ref) & Counter(can)
        num_same = sum(common.values())
        return num_same

    em_list = []
    for refs, can in zip(references, candidates):
        em_score = max([em(ref,can) for ref in refs])
        em_list.append(em_score)
        
    return {"exact_match": sum(em_list) /len(em_list)}

def f1_fn(references, candidates, word_level=False):
    def pre_recall_f1(reference,candidate):
        from collections import Counter
        common = Counter(reference) & Counter(candidate)
        num_same = sum(common.values()) #TP 
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(candidate)
        recall = 1.0 * num_same / len(reference)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    if word_level:
        reference = [" ".join(jieba.cut(ref)).split() for ref in references]
        candidate = [" ".join(jieba.cut(can)).split() for can in candidates]
    else:  # unigram token-level
        reference = [references[i][0] for i in range(len(references))]
        candidate = [candidates[i] for i in range(len(candidates))]

    prec, recall, f1 = [],[],[]
    for ref, can in zip(reference,candidate):
        (_pre, _re, _f1)  = pre_recall_f1(ref, can)
        prec.append(_pre)
        recall.append(_re)
        f1.append(_f1)

    return {
        "prec":mean(prec),
        "re":mean(recall),
        "f1":mean(f1)
    }

def acc_fn(references, candidates):
    num_same = 0 # 2-classification
    for ref, can in zip(references, candidates):
        ref = 0 if ref == "不检索" else 1
        can = 0 if can == "不检索" or "不" in can else 1
        if ref == can:
            num_same = num_same + 1
    
    return {"acc" :num_same / len(candidates)}     

def dist_fn(candidates):
    """
    Ref: https://github.com/microsoft/ProphetNet/blob/master/GLGE_baselines/script/script/evaluate/personachat/eval.py
    """
    batch_size = len(candidates)
    intra_dist1, intra_dist2 = [],[]
    unigrams_all, bigrams_all = Counter(), Counter()

    for can in candidates:
        unigrams = Counter(can)
        bigrams = Counter(zip(can,can[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(can)+1e-5)) 
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(can)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5) #所有回复中构成的词典 unigram 长度/长度
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = sum(intra_dist1) / len(intra_dist1) #每一句计算 dist 求均值 
    intra_dist2 = sum(intra_dist2) / len(intra_dist2)

    return {
        "dist_inter_1" : inter_dist1,
        "dist_inter_2" : inter_dist2,
        "dist_intra_1" : intra_dist1,
        "dist_intra_2" : intra_dist2,
    }     

def vocab_fn(candidates):
    vocab = []
    for can in candidates:
        counts = {}
        words = jieba.lcut(can)
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        vocab.append(len(counts.keys()))

    return {"vocab_length" : sum(vocab)/len(vocab)}

def load_huggingface_fn(metric):
    metric = evaluate.load(metric)
    return metric.compute
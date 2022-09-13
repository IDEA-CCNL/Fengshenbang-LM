import collections

import numpy as np

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False,
                                 geometric_dist=False,
                                 masking_style="bert",
                                 zh_tokenizer=None):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""
    '''
    modified from Megatron-LM
    Args:
        tokens: 输入
        vocab_id_list: 词表token_id_list
        vocab_id_to_token_dict： token_id到token字典
        masked_lm_prob：mask概率
        cls_id、sep_id、mask_id：特殊token
        max_predictions_per_seq：最大mask个数
        np_rng：mask随机数
        max_ngrams：最大词长度
        do_whole_word_mask：是否做全词掩码
        favor_longer_ngram：优先用长的词
        do_permutation：是否打乱
        geometric_dist：用np_rng.geometric做随机
        masking_style：mask类型
        zh_tokenizer：WWM的分词器，比如用jieba.lcut做分词之类的
    '''
    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)
    # 如果没有指定中文分词器，那就直接按##算
    if zh_tokenizer is None:
        for (i, token) in enumerate(tokens):
            if token == cls_id or token == sep_id:
                token_boundary[i] = 1
                continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
            if (do_whole_word_mask and len(cand_indexes) >= 1 and
                    not is_start_piece(vocab_id_to_token_dict[token])):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
                if is_start_piece(vocab_id_to_token_dict[token]):
                    token_boundary[i] = 1
    else:
        # 如果指定了中文分词器，那就先用分词器分词，然后再进行判断
        # 获取去掉CLS SEP的原始文本
        raw_tokens = []
        for t in tokens:
            if t != cls_id and t != sep_id:
                raw_tokens.append(t)
        raw_tokens = [vocab_id_to_token_dict[i] for i in raw_tokens]
        # 分词然后获取每次字开头的最长词的长度
        word_list = set(zh_tokenizer(''.join(raw_tokens), HMM=True))
        word_length_dict = {}
        for w in word_list:
            if len(w) < 1:
                continue
            if w[0] not in word_length_dict:
                word_length_dict[w[0]] = len(w)
            elif word_length_dict[w[0]] < len(w):
                word_length_dict[w[0]] = len(w)
        i = 0
        # 从词表里面检索
        while i < len(tokens):
            token_id = tokens[i]
            token = vocab_id_to_token_dict[token_id]
            if len(token) == 0 or token_id == cls_id or token_id == sep_id:
                token_boundary[i] = 1
                i += 1
                continue
            word_max_length = 1
            if token[0] in word_length_dict:
                word_max_length = word_length_dict[token[0]]
            j = 0
            word = ''
            word_end = i+1
            # 兼容以前##的形式，如果后面的词是##开头的，那么直接把后面的拼到前面当作一个词
            old_style = False
            while word_end < len(tokens) and vocab_id_to_token_dict[tokens[word_end]].startswith('##'):
                old_style = True
                word_end += 1
            if not old_style:
                while j < word_max_length and i+j < len(tokens):
                    cur_token = tokens[i+j]
                    word += vocab_id_to_token_dict[cur_token]
                    j += 1
                    if word in word_list:
                        word_end = i+j
            cand_indexes.append([p for p in range(i, word_end)])
            token_boundary[i] = 1
            i = word_end

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1. / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]
    # 获取一个ngram的idx，对于每个word，记录他的ngram的word
    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(ngrams[:len(cand_index_set)],
                              p=pvals[:len(cand_index_set)] /
                              pvals[:len(cand_index_set)].sum(keepdims=True))
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            token_id = tokens[index]
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=token_id))

        masked_spans.append(MaskedLmInstance(
            index=index_set,
            label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)

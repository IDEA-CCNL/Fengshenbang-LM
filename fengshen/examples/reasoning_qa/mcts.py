import random
import jsonlines
import math
import numpy as np
import time
import logging
import coloredlogs
from collections import defaultdict
from itertools import accumulate
import heapq
import gc
import sys
import os
from nltk import ngrams
import torch
import torch.nn.functional as F
from sample_sequence import top_k_logits, get_batch
import copy
from merge_data import text_filtering
from pysnooper import snoop
from torchsnooper import snoop as tsnoop
from calculator import use_calculator
from math_data_model import extract_answer, is_correct
logger = logging.getLogger('__file__')
coloredlogs.install(level='INFO', logger=logger)
EQUALS_TOKENS = set([28, 796, 47505])

"""
    A Monto Carlo Tree Search implementation for similar-text generation
"""

def pushq(listy, capacity, item):
    """item: (score, text)"""
    if item in listy:
        return

    if len(listy) < capacity:
        heapq.heappush(listy, item)
    else:
        heapq.heappushpop(listy, item)


class State():
    """
    This class represents the state of a node.
    param num_gen: length of tokens to be allowed to generate
    param is_terminal: whether a leaf node
    param token_ids: the token ids
    """

    def __init__(self, num_gen, token_ids, prob=1.0):
        self.num_gen = num_gen
        self.is_terminal = (self.num_gen == 0)
        self.token_ids = token_ids
        self.prob = prob

    def next(self, next_token_ids, prob):
        if self.num_gen <= 0:
            raise ValueError("exceed maximal allowed length")
        return State(self.num_gen-1, next_token_ids, prob)

    def __hash__(self):
        #  用token_ids是否相同来对比state是否相同，当expand时采出相同的child时重新采
        #  return self.token_ids
        return self.token_ids[0][0]  #  expand多步，只考虑第一个token如果已经出现，就不用同一个开头的token了

    def __eq__(self, others):
        return self.__hash__() == others.__hash__()


class Node(object):
    """
    This class defines the node of a search tree
    param visit: number of visit
    param parent: parent node
    param: state: state of current node
    param next_token_probs: prob. distributiion of next token
    param *mems: additional params in case of recomputing
    """

    def __init__(self, parent, state, next_token_probs, max_num_children, mems):
        self.visit = 0
        self.reward = 0.0
        self.parent = parent
        if self.parent is None:
            self.emitted_tokens = defaultdict(lambda:1)
        else:
            self.emitted_tokens = parent.emitted_tokens.copy()
            token_ids = state.token_ids.view(-1).tolist()
            for token in token_ids:
                self.emitted_tokens[token] += 1

        self.state = state
        self.children = []
        self.max_num_children = max_num_children # TODO:  vary gradually with the change of depth
        self.next_token_probs = next_token_probs
        self.mems = mems

    def __repr__(self):
        #  TODO 这里tokenizer用了全局变量，后续需要修改
        return f"token: {tokenizer.decode(self.state.token_ids.view(-1))}, visit: {self.visit}, reward: {self.reward}, prob: {self.state.prob}"
        #  , parent: {tokenizer.decode(self.parent.state.token_ids.view(-1))}"

    def add_child(self, child_state, child_next_token_probs, max_num_children, child_mems):
        child_node = Node(self, child_state, child_next_token_probs, max_num_children, child_mems)
        self.children.append(child_node)

    def update(self, reward, decay_rate = 0.95):
        self.visit += 1
        self.reward += decay_rate*reward

    def empty_cache(self):
        self.mems = None
        self.next_token_probs = None
        self.emitted_tokens = None
        # if hasattr(self, 'mems'):
        #     del self.mems
        # if hasattr(self, 'next_token_probs'):
        #     del self.next_token_probs
        # if hasattr(self, 'emitted_tokens'):
        #     del self.emitted_tokens
        torch.cuda.empty_cache()
        gc.collect()

    def is_fully_expanded(self):
        return len(self.children) == self.max_num_children

def add_common_ctrl_args(parser):
    "generation control"
    group = parser.add_argument_group('commn ctrl', 'configurations')
    #  group.add_argument("--model_dir", type=str, default='/cognitive_comp/wanghao/gts_generator/model/')
    #  group.add_argument("--input_dir", type=str, default='/cognitive_comp/wanghao/gts_generator/tests')
    #  group.add_argument("--input_file", type=str, default=None)
    #  group.add_argument("--input_key", type=str, default='content')
    #  group.add_argument("--label_key", type=str, default='label')
    #  group.add_argument("--output_dir", type=str, default='/cognitive_comp/wanghao/gts_generator/output')
    #  group.add_argument("--num_per_sample", type=int, default=None)
    #  group.add_argument("--out_seq_length", type=int, default=1024)
    #  group.add_argument("--max_allowed_time", type=float, default=None)
    #  group.add_argument("--max_gened_length", type=int, default=None)

    #  group.add_argument("--batch_size", type=int, default=10)
    group.add_argument("--device", type=str, default='cuda')
    #  group.add_argument("--method", type=int, default=None)
    group.add_argument("--temperature", type=float, default=1.1, help='sampling temperature')
    group.add_argument("--top_p", type=float, default=0.0)
    group.add_argument("--top_k", type=int, default=0)

    return parser

def add_mcts_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('mcts', 'configurations')

    group.add_argument("--max_num_children", type=int, default=4, help='maximal number of children of non-root node')
    group.add_argument("--root_max_num_children", type=int, default=10, help='maximal number of children of root node')
    group.add_argument("--roll_out_size", type=int, default=200, help='size of roll out.')
    group.add_argument("--sampling_size", type=int, default=1024, help='maximal sequence length allowed for sampling')
    group.add_argument("--max_length", type=int, default=1024, help='maximal depth of the tree')
    group.add_argument("--max_iter", type=int, default=500, help='maximally allowed iterations')
    group.add_argument("--sim_score_base", type=float, default=0.6, help='excepted lower bound of similary scores')
    group.add_argument("--time_out", type=float, default=180, help='maximally allowed runing time for each sample, in unit of second')
    group.add_argument("--alpha", type=float, default=1.0, help='balance similarity and fluency. Its value is within the range of [0, 1], `1` means ignore fluency while `0` means ignore similarity')
    group.add_argument("--c_out", type=float, default=10.0, help='coeffecient balacing exploration and expolitation')
    group.add_argument("--burn_in_rate", type=float, default=0.2, help='burn in the first few nodes')
    group.add_argument("--bp_decay_rate", type=float, default=1.0, help='decay rate during back propagation')
    group.add_argument("--select_strategy", type=MCTS.SELECT_STRATEGY, default=MCTS.SELECT_STRATEGY.BEST, help='select strategy')
    group.add_argument("--annealing_rate", type=float, default=10.0, help='annealing rate. mandatory if `select_strategy` is `MCTS.SELECT_STRATEGY.ANNEALING`')
    group.add_argument("--initialize_tree", dest='initialize_tree', action='store_true')
    group.add_argument("--use_cls", dest='use_cls', action='store_true')
    group.add_argument("--rep_penalty", type=float, default=0.1, help='penalize repetitions')
    group.add_argument("--ng", type=int, default=2, help='ngrams for calculating repetitions')
    group.add_argument("--sample_capacity", type=int, default=100)
    group.add_argument("--expand_length", type=int, default=1)
    group.add_argument("--split", type=str)
    group.add_argument("--expand_repeat_penalty", type=float, default=1.2)

    return parser


def add_gsm8k_args(parser):
    """GSM8k arguments."""

    group = parser.add_argument_group('gsm8k', 'configurations')

    group.add_argument("--verifier_type", type=str)
    group.add_argument("--verifier_name", type=str)
    group.add_argument("--expand_verifier_type", type=str)
    group.add_argument("--expand_verifier_name", type=str)
    group.add_argument("--model_name", type=str)
    group.add_argument("--data", type=str)
    group.add_argument("--timestamp", type=str)

    return parser


class MCTS():
    class SELECT_STRATEGY:
        RANDOM = 0
        ANNEALING = 1
        BEST = 2

    def __init__(self, model, tokenizer, args, device, verifier_model, verifier_head, verifier_tokenizer, expand_verifier_model, expand_verifier_head, expand_verifier_tokenizer, input_token_ids=None, scalar=1.0, label=None, root=None):
        self.model = model
        self.tokenizer = tokenizer
        self.thought_idx = tokenizer.convert_tokens_to_ids("[THOUGHT]")
        self.verifier_model = verifier_model
        self.verifier_head = verifier_head
        self.verifier_tokenizer = verifier_tokenizer
        self.verifier_idx = verifier_tokenizer.convert_tokens_to_ids("[VERIFIER]")
        self.expand_verifier_model = expand_verifier_model
        self.expand_verifier_head = expand_verifier_head
        self.expand_verifier_tokenizer = expand_verifier_tokenizer
        self.expand_verifier_idx = expand_verifier_tokenizer.convert_tokens_to_ids("[VERIFIER]")
        self.args = args
        self.device = device
        self.input_token_ids = input_token_ids

        assert input_token_ids is not None
        self.org_context_length = input_token_ids.size(1)
        self.max_num_gen = self.args.max_length - self.org_context_length
        self.eos_token = self.tokenizer.eos_token
        if args.sample_capacity < 0:
            self.sample_capacity = 2 * self.args.num_per_sample
        else:
            self.sample_capacity = args.sample_capacity
        self.scalar = scalar
        self.label = label
        self.good_cases = []
        # self.use_cls = (self.args.use_cls is True) and (self.label is not None)
        self.node_mem_len = 3

        if root is None:
            #  bos_token_id = self.tokenizer(["[QUES]"], return_tensors="pt").to(self.device).input_ids
            next_token_probs, mems = self.get_token_probs(index=0, token_ids=input_token_ids, node=None)
            self.root = Node(parent=None,
                             state=State(num_gen=self.max_num_gen, token_ids=input_token_ids.view(1, -1).to(self.device), prob=1.),
                             next_token_probs=next_token_probs,
                             max_num_children=args.root_max_num_children,
                             mems=mems,
                             )
        else:
            self.root = root

    #  @snoop()
    def search(self):
        if self.args.initialize_tree:
            logger.info('-'*20+'iter. 0' + '-'*20)
            self.initialize_tree_with_input()

        tic = time.time()
        for i in range(self.args.max_iter):
            #  logger.info('-'*20+f'iter. {i+1:4d}'+'-'*20)
            if time.time() - tic > self.args.time_out:
                #  logger.info('-'*20+f'iter. {i+1:4d}'+'-'*20)
                print('-'*20+f'iter. {i+1:4d}'+'-'*20)
                self.printTree()
                #  logger.info('-'*20 + f'time out: {self.args.time_out:4.0f} s' + '-'*20)
                print('-'*20 + f'time out: {self.args.time_out:4.0f} s' + '-'*20)
                return
            self.search_once()
        self.printTree()
        #  logger.info('-'*20 + f'maximal iterations reached: max_iter = {self.args.max_iter:4d}' + '-'*20)
        print('-'*20 + f'maximal iterations reached: max_iter = {self.args.max_iter:4d}' + '-'*20)

    #  @snoop()
    def search_once(self):
        front = self.search_policy(self.root)
        #  TODO 看一下node
        reward = self.reward(*self.roll_out(front, self.args.roll_out_size))
        self.back_prop(front, reward)
        #  print("New Node: ", front)
        gc.collect()
        torch.cuda.empty_cache()

    def initialize_tree_with_input(self):
        node = self.root
        for token_ids in self.input_token_ids:
            next_token = torch.LongTensor([[token_ids]]).to(self.device)
            child_state = node.state.next(next_token, (node.next_token_probs[0, next_token]).view(-1).mean().detach().cpu().tolist())
            # next_token_probs, *mems = self.get_token_probs(self.max_num_gen - child_state.num_gen, child_state.token_ids, node.emitted_tokens, *node.mems)
            next_token_probs, mems = self.get_token_probs(self.max_num_gen - child_state.num_gen, child_state.token_ids, node)
            # NOTE 大于一定深度的节点，不保存past_key_values，节省显存
            if self.max_num_gen - child_state.num_gen > self.node_mem_len:
                mems = None
            node.add_child(child_state, next_token_probs, self.args.max_num_children, mems)
            node = node.children[-1]

        #TODO 删除打印
        print("-" * 20 + "Whole Tree" + "-" * 20)
        self.printTree()
        reward = self.reward(*self.roll_out(node, self.args.roll_out_size))
        self.back_prop(node, reward)

    #  @snoop()
    def search_policy(self, node):
        # a hack to force 'exploitation'
        logger.debug("enter search_policy")
        last_token_id = node.state.token_ids.view(-1).tolist()
        #  last_token = self.tokenizer.convert_ids_to_tokens(last_token_id)
        # while (node.state.is_terminal is False) and (last_token[0] != self.args.eos_token) and ('”' not in eos_token):
        while (node.state.is_terminal is False) and (self.tokenizer.eos_token_id not in last_token_id):
            if len(node.children) == 0:
                #  return self.expand(node)
                #  return self.expand_with_calculator(node)
                return self.expand_multi_step_with_calculator(node)
            elif random.uniform(0, 1) < .5:
                # if node.is_fully_expanded():
                    # node.empty_cache()
                node = self.select(node, self.args.select_strategy)
            else:
                if node.is_fully_expanded() is False:
                    #  return self.expand(node)
                    #  return self.expand_with_calculator(node)
                    return self.expand_multi_step_with_calculator(node)
                else:
                    # node.empty_cache()
                    node = self.select(node, self.args.select_strategy)
            last_token_id = node.state.token_ids.view(-1).tolist()
            #  last_token = self.tokenizer.convert_ids_to_tokens(last_token_id)
        logger.debug("leave search_policy")
        return node

    #  @tsnoop()
    def expand_multi_step_with_calculator(self, node):
        logger.debug("enter expand multi step with calculator")
        already_tried = [c.state for c in node.children]
        next_token_probs = node.next_token_probs
        next_node_token_ids = []
        next_token_prob = []
        teriminal = False
        for i in range(1, self.args.expand_length + 1):
            if teriminal:
                break
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            #  multi step expand, 只考虑第一个token已经被sample过的情况
            if i == 1:
                tmp_state = node.state.next(next_token, 0)
                if tmp_state in already_tried and tmp_state.is_terminal is False:
                    # NOTE 不是直接置零，而是惩罚其概率，如果要置零，可以把expand_repeat_penalty设置得非常大，这样就等于置零了
                    #  next_token_probs[0, next_token] = 0.0 # suppress probs of states in already_tried
                    next_token_probs[0, next_token] /= self.args.expand_repeat_penalty # penalty probs of states in already_tried
                    #  print(f'next_token_probs {node.next_token_probs.shape}, sum {node.next_token_probs.sum(dim=1)}')
                    try:
                        next_token = torch.multinomial(next_token_probs, num_samples=1)
                        tmp_state = node.state.next(next_token, 0)
                    except: # no available candidates
                        node.max_num_children = len(node.children)
                        return node.children[already_tried.index(tmp_state)]

            prob = next_token_probs[0, next_token].detach().cpu().item()

            #  NOTE 如果expand的时候生成了<|endoftext|> break并且不再expand这个node
            if next_token.item() == self.tokenizer.eos_token_id:
                mems = None
                teriminal = True
                next_node_token_ids.extend(next_token.detach().cpu().view(-1).tolist())
                next_token_prob.append(prob)

            if next_token.item() in EQUALS_TOKENS:
                cur = node
                token_ids = cur.state.token_ids.detach().cpu().view(-1).tolist() + next_token.detach().cpu().view(-1).tolist()
                while cur.parent is not None:
                    #  token_ids = torch.cat((cur.parent.state.token_ids, token_ids), dim=-1)
                    token_ids = cur.parent.state.token_ids.detach().cpu().view(-1).tolist() + token_ids
                    cur = cur.parent
                text = tokenizer.decode(token_ids)
                answer = use_calculator(text)
                if answer is not None:
                    #  print(text)
                    #  print(answer)
                    text = text + str(answer) + ">>"
                    next_token = torch.cat((next_token, tokenizer([str(answer) + ">>"], return_tensors="pt").to(self.device).input_ids), dim=-1)
                else:
                    logger.error(f"= generated but no answer is got, text: {text}, token_ids: {token_ids}.")

            next_node_token_ids.extend(next_token.detach().cpu().view(-1).tolist())
            next_token_prob.append(prob)

            #  NOTE get_token_probs 的 index 除了0以外都一样，没什么用
            next_token_probs, mems = self.get_token_probs(-1, torch.tensor(next_node_token_ids, device=self.device).view(1, -1), node)


        child_state = State(node.state.num_gen - len(next_node_token_ids), torch.tensor(next_node_token_ids, device=self.device).view(1, -1), np.mean(next_token_prob))
        child_state.is_terminal = teriminal or child_state.is_terminal

        if node.state.num_gen - child_state.num_gen > self.node_mem_len:
            mems = None
        node.add_child(child_state, next_token_probs, self.args.max_num_children, mems)

        logger.debug("leave expand")

        # -----------------calculate expand reward---------------------
        cur = node.children[-1]
        logsum = [math.log(prob+1.0e-20) for prob in next_token_prob]
        ippl = math.exp(sum(logsum) / len(logsum)) # inverse perplexity
        path_tokens = next_node_token_ids[:]
        while cur.parent is not None:
            node_token = cur.parent.state.token_ids.detach().cpu().view(-1).tolist()
            path_tokens = node_token + path_tokens
            cur = cur.parent
        reward = self.expand_reward(path_tokens, ippl)
        self.back_prop(node.children[-1], reward)
        gc.collect()
        torch.cuda.empty_cache()
        # -------------------------end---------------------------------
        return node.children[-1]

    #  @snoop()
    #  def expand_with_calculator(self, node):
    #      logger.debug("enter expand with calculator")
    #      already_tried = [c.state for c in node.children]
    #
    #      next_token = torch.multinomial(node.next_token_probs, num_samples=1)
    #      child_state = node.state.next(next_token, (node.next_token_probs[0, next_token]).view(-1).mean().detach().cpu().tolist())
    #      while child_state in already_tried and child_state.is_terminal is False:
    #          node.next_token_probs[0, next_token] = 0.0 # suppress probs of states in already_tried
    #          #  print(f'next_token_probs {node.next_token_probs.shape}, sum {node.next_token_probs.sum(dim=1)}')
    #          try:
    #              next_token = torch.multinomial(node.next_token_probs, num_samples=1)
    #          except: # no available candidates
    #              node.max_num_children = len(node.children)
    #              return node.children[already_tried.index(child_state)]
    #          child_state = node.state.next(next_token, (node.next_token_probs[0, next_token]).view(-1).mean().detach().cpu().tolist())
    #
    #      next_token_prob = (node.next_token_probs[0, next_token]).view(-1).mean().detach().cpu().tolist()
    #      if next_token.item() in EQUALS_TOKENS:
    #          cur = node
    #          # TODO .view(-1)展开后变为list
    #          token_ids = cur.state.token_ids.detach().cpu().view(-1).tolist() + next_token.detach().cpu().view(-1).tolist()
    #          while cur.parent is not None:
    #              #  token_ids = torch.cat((cur.parent.state.token_ids, token_ids), dim=-1)
    #              token_ids = cur.parent.state.token_ids.detach().cpu().view(-1).tolist() + token_ids
    #              cur = cur.parent
    #          text = tokenizer.decode(token_ids)
    #          answer = use_calculator(text)
    #          if answer is not None:
    #              text = text + str(answer) + ">>"
    #              next_token = torch.cat((next_token, tokenizer([str(answer) + ">>"], return_tensors="pt").to(self.device).input_ids), dim=-1)
    #      child_state = node.state.next(next_token, next_token_prob)
    #
    #      #  next_token_probs, *mems = self.get_token_probs(self.max_num_gen - node.state.num_gen, node.state.token_ids, node.emitted_tokens, *node.mems)
    #      next_token_probs, mems = self.get_token_probs(self.max_num_gen - child_state.num_gen, child_state.token_ids, node)
    #
    #      #  NOTE 因为root节点的token_ids是question，长度比较长，所以这里用root的num_gen-child的num_gen作为最大保留past_key_values的深度
    #      #  if self.max_num_gen - child_state.num_gen > self.node_mem_len:
    #      if node.state.num_gen - child_state.num_gen > self.node_mem_len:
    #          mems = None
    #      node.add_child(child_state, next_token_probs, self.args.max_num_children, mems)
    #      logger.debug("leave expand")
    #      return node.children[-1]

    #  @snoop()
    #  def expand(self, node):
    #      logger.debug("enter expand")
    #      already_tried = [c.state for c in node.children]
    #
    #      next_token = torch.multinomial(node.next_token_probs, num_samples=1)
    #      child_state = node.state.next(next_token, (node.next_token_probs[0, next_token]).view(-1).mean().detach().cpu().tolist())
    #      while child_state in already_tried and child_state.is_terminal is False:
    #          node.next_token_probs[0, next_token] = 0.0 # suppress probs of states in already_tried
    #          #  print(f'next_token_probs {node.next_token_probs.shape}, sum {node.next_token_probs.sum(dim=1)}')
    #          try:
    #              next_token = torch.multinomial(node.next_token_probs, num_samples=1)
    #          except: # no available candidates
    #              node.max_num_children = len(node.children)
    #              return node.children[already_tried.index(child_state)]
    #          child_state = node.state.next(next_token, (node.next_token_probs[0, next_token]).view(-1).mean().detach().cpu().tolist())
    #
    #      #  next_token_probs, *mems = self.get_token_probs(self.max_num_gen - node.state.num_gen, node.state.token_ids, node.emitted_tokens, *node.mems)
    #      next_token_probs, mems = self.get_token_probs(self.max_num_gen - child_state.num_gen, child_state.token_ids, node)
    #
    #      #  NOTE 因为root节点的token_ids是question，长度比较长，所以这里用root的num_gen-child的num_gen作为最大保留past_key_values的深度
    #      #  if self.max_num_gen - child_state.num_gen > self.node_mem_len:
    #      if node.state.num_gen - child_state.num_gen > self.node_mem_len:
    #          mems = None
    #      node.add_child(child_state, next_token_probs, self.args.max_num_children, mems)
    #      logger.debug("leave expand")
    #      return node.children[-1]

    #  @snoop()
    def select(self, node, select_strategy=SELECT_STRATEGY.ANNEALING):
        logger.debug("enter select")

        best_node = None
        if select_strategy == MCTS.SELECT_STRATEGY.BEST:
            best_score = float("-inf")
            best_nodes = []

            for c in node.children:
                score,_,_ = self.calc_ee_score(c, node)
                if score == best_score:
                    best_nodes.append(c)
                elif score > best_score:
                    best_nodes = [c]
                    best_score = score

            best_node = random.choice(best_nodes)

        elif select_strategy == MCTS.SELECT_STRATEGY.ANNEALING:
            score = []
            for c in node.children:
                score.append(self.calc_ee_score(c, node)[0])

            score = torch.tensor(score)
            probs = torch.softmax(score*self.args.annealing_rate*(self.max_num_gen - node.state.num_gen + 1), dim=0) # quite flat at first few nodes, then more focusing
            idx = torch.multinomial(probs, 1)[0].tolist()
            best_node = node.children[idx]
        else:
            score = []
            for c in node.children:
                score.append(self.calc_ee_score(c, node)[0])
                probs = np.array(score)
                probs /= np.sum(probs)

            idx = list(np.random.multinomial(1, probs)).index(1)  # TODO 为什么这里index(1)?
            best_node = node.children[idx]

        logger.debug("leave select")
        return best_node

#      def random_select(self, node):
#          score = []
#          for c in node.children:
#              score.append(self.calc_ee_score(c, node)[0])
#          score = np.array(score)
#          score /= score.sum()
#  #         idx = np.random.choice(len(node.children), 1, p=score)[0]
#          idx = list(np.random.multinomial(1, score)).index(1)
#          return node.children[idx]

    #  @tsnoop()
    def roll_out(self, node, roll_out_size):
        logger.debug("enter roll_out")

        path_tokens = []
        logsum = []
        def update_tokens(tokens, prob, num_tokens, append=True):
            for i, t in enumerate(tokens):
                num_tokens += 1
                if append:
                    path_tokens.append(t)
                    if i == 0:
                        logsum.append(math.log(prob+1.0e-20))
                    # TODO 什么时候会出现走到这里的情况？
                    else:
                        logsum.append(0.0)
                else:
                    path_tokens.insert(i, t)
                    if i == 0:
                        logsum.insert(0, math.log(prob+1.0e-20))
                    # TODO 什么时候会出现走到这里的情况？
                    else:
                        logsum.insert(i, 0.0)
            return num_tokens

        num_tokens = 0
        node_token = node.state.token_ids.detach().cpu().view(-1).tolist()
        num_tokens = update_tokens(node_token, node.state.prob, num_tokens)

        # history tokens
        cur = node
        while cur.parent is not None:
            node_token = cur.parent.state.token_ids.detach().cpu().view(-1).tolist()
            num_tokens = update_tokens(node_token, cur.parent.state.prob, num_tokens, append=False)
            # path_tokens.insert(0, cur.parent.state.token_ids.detach().cpu().view(-1).tolist()[0])
            # if cur.parent.state.prob > 1.0e-20:
            #     logsum.insert(0, math.log(cur.parent.state.prob))
            #     num_tokens += 1
            cur = cur.parent

        cur = node
        while not hasattr(cur, 'mems'):
        #  while cur.mems is None:
            # cur = cur.children[-1]
            # path_tokens.extend(cur.state.token_ids.detach().cpu().view(-1).tolist())
            cur = cur.children[-1]
            node_token = cur.state.token_ids.detach().cpu().view(-1).tolist()
            num_tokens = update_tokens(node_token, cur.state.prob, num_tokens)

        # mems = cur.mems
        # state = cur.state
        # emitted_tokens = cur.emitted_tokens.copy()
        tmp_node = copy.deepcopy(cur)

        tmp_node_token_ids = tmp_node.state.token_ids.view(-1).tolist()
        gen_len = 0
        # while (tmp_node.state.is_terminal is False) and (self.args.eod_token not in eod) and ('”' not in eod_token) and (gen_len < roll_out_size):
        while (tmp_node.state.is_terminal is False) and (self.tokenizer.eos_token_id not in tmp_node_token_ids) and (gen_len < roll_out_size):
            gen_len += 1
            index = self.max_num_gen - tmp_node.state.num_gen
            # probs, *mems = self.get_token_probs(index, state.token_ids, emitted_tokens, *mems)
            probs, mems = self.get_token_probs(index, tmp_node.state.token_ids, tmp_node)
            tmp_node.mems = mems
            next_token = torch.multinomial(probs, num_samples=1)
            prob = (probs[0, next_token]).view(-1).mean().detach().cpu().tolist()

            if next_token.item() in EQUALS_TOKENS:
                #  token_ids = torch.cat((tmp_node.state.token_ids, next_token), dim=-1).view(-1)
                token_ids = path_tokens + [next_token.detach().cpu().item()]
                token_ids = list(filter(lambda x: x is not None, token_ids))
                assert None not in token_ids, f"{token_ids}"
                try:
                    text = tokenizer.decode(token_ids)
                except:
                    logger.error("decode bugs")
                    print(tmp_node)
                    self.printTree()
                    print("length of token_ids: ", len(token_ids), token_ids)
                    raise ValueError()
                answer = use_calculator(text)
                if answer is not None:
                    text = text + str(answer) + ">>"
                    next_token = torch.cat((next_token, tokenizer([str(answer) + ">>"], return_tensors="pt").to(self.device).input_ids), dim=-1)

            tmp_node.state = tmp_node.state.next(next_token, prob)
            tmp_node_token_ids = tmp_node.state.token_ids.view(-1).tolist()

            node_token = next_token.detach().cpu().view(-1).tolist()
            num_tokens = update_tokens(node_token, tmp_node.state.prob, num_tokens)
            # path_tokens.extend(next_token.detach().cpu().view(-1).tolist())
            # logsum.append(math.log(state.prob))
            # num_tokens += 1

            # NOTE 不需要, 注释
            # update frequencies of emitted tokens 
            #  new_token_ids = next_token.view(-1).tolist()
            #  for new_token in new_token_ids:
            #      tmp_node.emitted_tokens[new_token] += 1

        #  NOTE 不需要删除开始的一些可能影响流畅度的token
        #  cumsum = list(accumulate(logsum))
        #  burned_len = len(list(filter(lambda x: x >= cumsum[-1]*self.args.burn_in_rate, cumsum)))
        #  print(f'logsum {logsum}, cumsum {cumsum}, rate {cumsum[-1]*(1 - self.args.burn_in_rate)}')

        #  ippl = math.exp(sum(logsum[burned_len:])/float(num_tokens - burned_len)) # inverse perplexity
        #  path_tokens = path_tokens[burned_len:]
        #  logsum = logsum[burned_len:]
        ippl = math.exp(sum(logsum) / num_tokens) # inverse perplexity

        logger.debug("leave roll_out")
        #  print("path tokens:", self.tokenizer.decode(path_tokens))

        return path_tokens, ippl

    #  @snoop()
    def back_prop(self, node, reward):
        logger.debug("enter back_prop")

        cur = node
        decay_rate = self.args.bp_decay_rate
        while cur is not None:
            cur.update(reward, decay_rate)
            cur = cur.parent
            decay_rate *= decay_rate
        logger.debug("leave back_prop")

    #  @snoop()
    #  def sampling(self, sel_func, *args, **kwargs):
    #      node = self.root
    #      while node.state.is_terminal is False and len(node.children) > 0:
    #          node = sel_func(node, *args, **kwargs)
    #
    #      tokens = self.roll_out(node, self.args.sampling_size)[0]
    #      tokens = tokens[1:] if tokens[0] == 0 else tokens
    #
    #      output_text = self.tokenizer.convert_tokens_to_ids(tokens)
    #      output_text = text_filtering(output_text.split(self.eos_token)[0])
    #
    #      if not self.use_cls:
    #          score = self.calc_sim_score(output_text)
    #      else:
    #          score = self.calc_cls_score(output_text)
    #
    #      if self.args.rep_penalty > 0.0 and len(self.good_cases) > 0:
    #          score = (1-self.args.rep_penalty)*score - self.args.rep_penalty*self.calc_repetition_score(output_text)
    #
    #      logger.info(f'sampling: {output_text}, {self.label}, {score}')
    #      return score, output_text

    def expand_reward(self, tokens, fluency=1.0):
        output_ids = tokens
        output_text = self.tokenizer.decode(output_ids)
        output_text = output_text.replace(" [ANS] ", "[ANS]")
        #  NOTE 这里是为了bert的token_type_ids, 把question和thought用[SEP]分隔开来。
        q_and_t = output_text.split("[THOUGHT]", maxsplit=1)
        #  TODO 暂时不做这样额外的人工规则修改
        #  防止生成了多个[THOUGHT], 这里把最后一个[THOUGHT]后面的内容都当做生成的思考
        #  thought = "[THOUGHT]" + q_and_t[1].split("[THOUGHT]")[-1]
        ques, thought = q_and_t[0], "[THOUGHT]" + q_and_t[1]
        if self.args.expand_verifier_type == "bert" or self.args.expand_verifier_type == "deberta":
            inputs_encoding = self.expand_verifier_tokenizer([ques], [thought], return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512).to(self.device)
        elif self.args.expand_verifier_type == "gpt":
            inputs_encoding = self.expand_verifier_tokenizer(ques + thought, return_tensors="pt", add_special_tokens=False).to(self.device)

        verifier_score = self.calc_expand_verifier_score(**inputs_encoding)

        #  if self.args.alpha < 1.0:
        #      score = math.pow(verifier_score, self.args.alpha) * math.pow(fluency, 1.0 - self.args.alpha) # strenthen languate model if alpha < 1
        score = verifier_score

        return score

    #  @snoop()
    def reward(self, tokens, fluency=1.0):
        output_ids = tokens
        output_text = self.tokenizer.decode(output_ids)
        if self.tokenizer.eos_token_id not in output_ids:
            logger.warning(f"No eos token! {output_text}")
        # NOTE 保留eos token，方便之后正则抽取时有多个[ANS]不知道答案到底在哪，这样可以抽取[ANS]和<|endoftext|>中间的作为答案
        #  output_text = output_text.split(self.eos_token)[0]
        output_text = output_text.replace(" [ANS] ", "[ANS]")
        #  NOTE 这里是为了bert的token_type_ids, 以及防止生成了多个[THOUGHT], 这里只把第一个[THOUGHT]，即question末尾的[THOUGHT]后面的内容都当做生成的思考
        q_and_t = output_text.split("[THOUGHT]", maxsplit=1)
        #  if len(q_and_t) > 2:
        #      logger.warning(f"More than one [THOUGHT] token! {output_text}")
        #  TODO 暂时不做这样额外的人工规则修改
        #  防止生成了多个[THOUGHT], 这里把最后一个[THOUGHT]后面的内容都当做生成的思考
        #  thought = "[THOUGHT]" + q_and_t[1].split("[THOUGHT]")[-1]
        ques, thought = q_and_t[0], "[THOUGHT]" + q_and_t[1]
        #  ques, thought = q_and_t[0], "[THOUGHT]" + q_and_t[1].replace("[THOUGHT]", "")  # NOTE 防止出现多个[THOUGHT]
        if self.args.verifier_type == "bert" or self.args.verifier_type == "deberta":
            inputs_encoding = self.verifier_tokenizer([ques], [thought], return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512).to(self.device)
        elif self.args.verifier_type == "gpt":
            inputs_encoding = self.verifier_tokenizer(ques + thought, return_tensors="pt", add_special_tokens=False).to(self.device)
        #  NOTE 我似乎并不需要这个text_filtering
        #  output_text = text_filtering(output_text.split(self.eos_token)[0])

        #  TODO 这里加入一个calc_verifier_score函数替代了calc_sim和calc_cls
        verifier_score = self.calc_verifier_score(**inputs_encoding)

        #  NOTE 我不需要和关心重复ngram
        #  if self.args.rep_penalty > 0.0 and len(self.good_cases) > 0:
        #      score = (1-self.args.rep_penalty)*score - self.args.rep_penalty*self.calc_repetition_score(output_text)

        if self.args.alpha < 1.0:
            score = math.pow(verifier_score, self.args.alpha) * math.pow(fluency, 1.0 - self.args.alpha) # strenthen languate model if alpha < 1
        else:
            score = verifier_score

        #  TODO 暂时全部保留
        pushq(self.good_cases, self.sample_capacity, (score, thought))
        #  if verifier_score > 0.5:
            #  print(f'Good Case: verifier score={verifier_score}, fluency={fluency}, final_score={score}\n{output_text}', flush=True)
            #  pushq(self.good_cases, self.sample_capacity, (score, output_text))
        #  else:
            #  print(f'Bad Case: verifier score={verifier_score}, fluency={fluency}, final_score={score}\n{output_text}', flush=True)

        return score

    #  def calc_sim_score(self, output_tokens):
    #      try:
    #          return self.scorer.eval([self.input_tokens], [output_tokens])[-1]
    #      except:
    #          return 0.0
    #
    #  def calc_cls_score(self, tokens):
    #      return self.scorer.predict(tokens, self.label)

    def calc_expand_verifier_score(self, **inputs_encoding):
        output = self.expand_verifier_model(**inputs_encoding)
        # Bert 取第一个token(cls)，GPT取最后一个token
        verifier_logits = output.logits[:, 0 if self.args.expand_verifier_type == "bert" or self.args.expand_verifier_type == "deberta" else -1, self.expand_verifier_idx].half()  # Expected shape = (bs, )
        #  bs = 1, 非batch不需要这步
        #  verifier_logits = torch.gather(verifier_logits, 1, final_token_idx)  # Expected shape = (bs, 1)
        verifier_predictions = self.expand_verifier_head(verifier_logits.unsqueeze(-1))  # Expected shape = ()

        return verifier_predictions.item()

    def calc_verifier_score(self, **inputs_encoding):
        output = self.verifier_model(**inputs_encoding)
        # Bert 取第一个token(cls)，GPT取最后一个token
        verifier_logits = output.logits[:, 0 if self.args.verifier_type == "bert" or self.args.verifier_type == "deberta" else -1, self.verifier_idx].half()  # Expected shape = (bs, )
        #  bs = 1, 非batch不需要这步
        #  verifier_logits = torch.gather(verifier_logits, 1, final_token_idx)  # Expected shape = (bs, 1)
        verifier_predictions = self.verifier_head(verifier_logits.unsqueeze(-1))  # Expected shape = ()

        return verifier_predictions.item()

    #  def calc_repetition_score(self, tokens):
    #      if len(self.good_cases) == 0:
    #          return 0.0
    #      ngram_set = []
    #      for _,text in self.good_cases:
    #          grams = set()
    #          n_grams = ngrams(text, self.args.ng)
    #          for gram in n_grams:
    #              grams.add(''.join(gram))
    #          ngram_set.append(grams)
    #      candit_set = set()
    #      n_grams = ngrams(tokens, self.args.ng)
    #      for gram in n_grams:
    #          candit_set.add(''.join(gram))
    #
    #      costs = 0.0
    #      for i in range(len(self.good_cases)):
    #          costs += len(ngram_set[i].intersection(candit_set))/(min(len(ngram_set[i]), len(candit_set)) + 1)
    #      return costs/len(self.good_cases)

    #  @snoop()
    def calc_ee_score(self, node, parent):
        exploit = node.reward / max(node.visit, 1)
        explore = math.sqrt(float(parent.visit))/float(1 + node.visit)

        explore = self.scalar * explore * node.state.prob
        #  TODO 暂时不使用explore
        #  return exploit + explore, exploit, explore
        return exploit, exploit, explore

    def printTree(self, sep=8, start_node=None):
        def _dfs(parent, level=0):
            if level > 0:
                strs = ''
                for i in range(level-1):
                    strs += '|' + ' '*sep
                strs += '|->'
                token = self.tokenizer.convert_ids_to_tokens(parent.state.token_ids.view(-1).detach().cpu().tolist())
                score, exploit, explore = self.calc_ee_score(parent, parent.parent)
                #  print(f'{strs}{token}(score:{score:.2f},exploit:{exploit:.2f},explore:{explore:.2f})')
                print(f'{strs}{token}({score:.2f},{exploit:.2f},{explore:.2f})', flush=True)
            for node in parent.children:
                _dfs(node, level + 1)

        if start_node is None:
            _dfs(self.root)
        else:
            _dfs(start_node)

    #  @snoop()
    def traverse(self):
        def recursive(node, tokens, total):
            sent = tokens.copy()
            sent.append(node.state.token_ids.view(1).detach().cpu().tolist()[0])

            if node.state.is_terminal or len(node.children) == 0:
                total.append(self.tokenizer.convert_tokens_to_ids(sent[1:]))
                return
            for c in node.children:
                recursive(c, sent, total)

        full_path = []
        recursive(self.root, [], full_path)
        return full_path

    # def get_token_probs(self, index, tokens, emitted_tokens, *mems):
    #  @tsnoop()
    def get_token_probs(self, index, token_ids, node):
        self.model.eval()
        #  inputs_encoding = self.tokenizer(text, return_attention_mask=True, return_tensors="pt", padding=False).to(self.device)
        with torch.no_grad():
            if index == 0:
                mems = None
                #  tokens, attention_mask, position_ids = get_batch(self.context_tokens_tensor, self.device, self.args, batch_size=1)
                #  logits, *new_mems = self.model(**inputs_encoding, return_dict=True)
            else:
                cur = node
                mems = cur.mems
                #  NOTE 这里一直往上找，不断地把父节点的token_ids concat到当前token_ids前面，直到找到有past_key_values的节点，然后利用它的past_key_values接着往下生成
                while mems is None:
                    cur = cur.parent
                    token_ids = torch.cat((cur.state.token_ids, token_ids), dim=-1)
                    mems = cur.mems

            outputs = self.model(token_ids, past_key_values=mems, return_dict=True)

        logits = outputs.logits
        # TODO 改成了取最后一个token的logits, NOTE gpt-j里config的vocab size是50400，但toeknizer的vocab size只有50250+
        logits = logits[:, -1, :self.tokenizer.vocab_size + len(self.tokenizer.added_tokens_decoder)]
        # 将[THOUGHT]的概率置零
        logits[:, self.thought_idx] = -100
        new_mems = outputs.past_key_values

        emitted_tokens = node.emitted_tokens if node is not None else {}

        # NOTE 我不需要这个重复token惩罚，注释掉
        #  index = torch.LongTensor(list(emitted_tokens.keys())).view(1, -1).to(logits.device)
        #  values = torch.tensor(list(emitted_tokens.values()), dtype=logits.dtype).view(1, -1).to(logits.device)
        indicator = torch.ones_like(logits) * self.args.temperature

        logits /= indicator
        logits = top_k_logits(logits, top_k=self.args.top_k, top_p=self.args.top_p)
        next_token_probs = F.softmax(logits, dim=-1)

        return next_token_probs, new_mems

def load_verifier(verifier_type, verifier_name):
    if verifier_type == "bert":
        verifier_model = AutoModelForMaskedLM.from_pretrained(verifier_name)
        verifier_tokenizer = BertTokenizer.from_pretrained(verifier_name)
    elif verifier_type == "gpt":
        verifier_model = AutoModelForCausalLM.from_pretrained(verifier_name)
        verifier_tokenizer = GPT2Tokenizer.from_pretrained(verifier_name)
    elif verifier_type == "deberta":
        verifier_model = DebertaV2ForMaskedLM.from_pretrained(verifier_name)
        verifier_tokenizer = DebertaV2Tokenizer.from_pretrained(verifier_name, use_fast=True)
    verifier_head = torch.load(os.path.join(verifier_name, "verifier_head.pth"))

    return verifier_model.half().to(device), verifier_tokenizer, verifier_head.half().to(device)

def main(parser):
    parser = add_common_ctrl_args(parser)
    parser = add_mcts_args(parser)
    args = parser.parse_args()
    print(vars(args))

    data = DataProcessor._read_jsonl(args.data)
    for ex in data:
        ex['answer'] += "<|endoftext|>"
    #  {"question": "[QUES]Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n", "ground_truth": "[THOUGHT]Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n[ANS] 18", "solution": "[THOUGHT]Janet harvests 16 * 3 = <<16*3=48>>48 eggs every day for breakfast.\nShe harvests 48 * 4 = <<48*4=192>>192 muffins for her friends.\nShe sells 192 * 2 = <<192*2=384>>384 dollars for those muffins.\nThus, she makes 384 * $2 = $<<384*2=768>>768 at the farmers� market every day.\n[ANS] 768", "is_correct": false, "question_id": "0"}
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.half().to(device)
    # --------------------------roll out verifier-------------------------------
    verifier_model, verifier_tokenizer, verifier_head = load_verifier(args.verifier_type, args.verifier_name)
    # --------------------------expand verifier-------------------------------
    expand_verifier_model, expand_verifier_tokenizer, expand_verifier_head = load_verifier(args.expand_verifier_type, args.expand_verifier_name)
    #  if model.config.vocab_size < len(tokenizer):
    #      model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    with jsonlines.open(os.path.join(args.model_name, args.timestamp + "-mcts_verifier_file.jsonl" + args.split), 'a', flush=True) as f:
        for idx, sample in enumerate(data):
            question = "[QUES]" + sample['question'] + "\n[THOUGHT]"
            answer = sample['answer'].replace("####", "[ANS]")
            ground_truth = extract_answer(answer)
            #  print("ground_truth", ground_truth)

            #  question = "[QUES]Two girls each got 1/6 of the 24 liters of water. Then a boy got 6 liters of water. How many liters of water were left?\n[THOUGHT]"
            input_token_ids = tokenizer(question, return_tensors="pt").to(device).input_ids.view(1, -1)

            mcts = MCTS(model, tokenizer, args, device, verifier_model, verifier_head, verifier_tokenizer, expand_verifier_model, expand_verifier_head, expand_verifier_tokenizer, input_token_ids=input_token_ids, scalar=1.0)
            mcts.search()
            sample['question'] = "[QUES]" + sample['question'] + "\n"
            sample['ground_truth'] = "[THOUGHT]" + sample['answer'].replace("####", "[ANS]")
            del sample['answer']

            for case in mcts.good_cases:
                prediction = extract_answer(case[1])
                #  print("predicition", prediction)
                score = case[0]
                if ground_truth == prediction:
                    print(f"Question {sample['question_id']}, correct prediction: ", case)
                f.write({**sample, "solution": case[1], "verifier_score": str(score), "is_correct": ground_truth == prediction, })


if __name__ == "__main__":
    import argparse
    from data_preprocess import DataProcessor
    from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForMaskedLM, AutoTokenizer, BertTokenizer, DebertaV2Tokenizer, DebertaV2ForMaskedLM
    parser = argparse.ArgumentParser()
    parser = add_gsm8k_args(parser)
    args, argv = parser.parse_known_args()
    device = "cuda:0"
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    main(parser)


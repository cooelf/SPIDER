from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
import os
from random import randrange
import random
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import json
import collections
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


# def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
#     """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
#     with several refactors to clean it up and remove a lot of unnecessary variables."""
#     cand_indices = []
#     for (i, token) in enumerate(tokens):
#         if token == "[CLS]" or token == "[SEP]":
#             continue
#         cand_indices.append(i)
#
#     num_to_mask = min(max_predictions_per_seq,
#                       max(1, int(round(len(tokens) * masked_lm_prob))))
#     shuffle(cand_indices)
#     mask_indices = sorted(sample(cand_indices, num_to_mask))
#     masked_token_labels = []
#     for index in mask_indices:
#         # 80% of the time, replace with [MASK]
#         if random() < 0.8:
#             masked_token = "[MASK]"
#         else:
#             # 10% of the time, keep original
#             if random() < 0.5:
#                 masked_token = tokens[index]
#             # 10% of the time, replace with random word
#             else:
#                 masked_token = choice(vocab_list)
#         masked_token_labels.append(tokens[index])
#         # Once we've saved the true label for that token, we can overwrite it with the masked version
#         tokens[index] = masked_token
#
#     return tokens, mask_indices, masked_token_labels


def create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""

    # n-gram masking Albert
    ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        cand_indices.append(i)
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)
    masked_token_labels = []
    covered_indices = set()
    for index in cand_indices:
        n = np.random.choice(ngrams, p=pvals)
        if len(masked_token_labels) >= num_to_mask:
            break
        if index in covered_indices:
            continue
        if index < len(cand_indices) - (n - 1):
            for i in range(n):
                ind = index + i
                if ind in covered_indices:
                    continue
                covered_indices.add(ind)
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[ind]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = random.choice(vocab_list)
                masked_token_labels.append(MaskedLmInstance(index=ind, label=tokens[ind]))
                tokens[ind] = masked_token

    #assert len(masked_token_labels) <= num_to_mask
    masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_token_labels]
    masked_labels = [p.label for p in masked_token_labels]
    return tokens, mask_indices, masked_labels

def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                   max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_words):
    """Creates `TrainingInstance`s for a single document.
     This method is changed to create sentence-order prediction (SOP) followed by idea from paper of ALBERT, 2019-08-28, brightmart
    """
    document = all_documents[document_index]  # 得到一个文档

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:  # 有一定的比例，如10%的概率，我们使用比较短的序列长度，以缓解预训练的长序列和调优阶段（可能的）短序列的不一致情况
        target_seq_length = random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    # 设法使用实际的句子，而不是任意的截断句子，从而更好的构造句子连贯性预测的任务
    instances = []
    current_chunk = []  # 当前处理的文本段，包含多个句子
    current_length = 0
    i = 0
    # print("###document:",document) # 一个document可以是一整篇文章、新闻、词条等. document:[['是', '爷', '们', '，', '就', '得', '给', '媳', '妇', '幸', '福'], ['关', '注', '【', '晨', '曦', '教', '育', '】', '，', '获', '取', '育', '儿', '的', '智', '慧', '，', '与', '孩', '子', '一', '同', '成', '长', '！'], ['方', '法', ':', '打', '开', '微', '信', '→', '添', '加', '朋', '友', '→', '搜', '号', '→', '##he', '##bc', '##x', '##jy', '##→', '关', '注', '!', '我', '是', '一', '个', '爷', '们', '，', '孝', '顺', '是', '做', '人', '的', '第', '一', '准', '则', '。'], ['甭', '管', '小', '时', '候', '怎', '么', '跟', '家', '长', '犯', '混', '蛋', '，', '长', '大', '了', '，', '就', '底', '报', '答', '父', '母', '，', '以', '后', '我', '媳', '妇', '也', '必', '须', '孝', '顺', '。'], ['我', '是', '一', '个', '爷', '们', '，', '可', '以', '花', '心', '，', '可', '以', '好', '玩', '。'], ['但', '我', '一', '定', '会', '找', '一', '个', '管', '的', '住', '我', '的', '女', '人', '，', '和', '我', '一', '起', '生', '活', '。'], ['28', '岁', '以', '前', '在', '怎', '么', '玩', '都', '行', '，', '但', '我', '最', '后', '一', '定', '会', '找', '一', '个', '勤', '俭', '持', '家', '的', '女', '人', '。'], ['我', '是', '一', '爷', '们', '，', '我', '不', '会', '让', '自', '己', '的', '女', '人', '受', '一', '点', '委', '屈', '，', '每', '次', '把', '她', '抱', '在', '怀', '里', '，', '看', '她', '洋', '溢', '着', '幸', '福', '的', '脸', '，', '我', '都', '会', '引', '以', '为', '傲', '，', '这', '特', '么', '就', '是', '我', '的', '女', '人', '。'], ['我', '是', '一', '爷', '们', '，', '干', '什', '么', '也', '不', '能', '忘', '了', '自', '己', '媳', '妇', '，', '就', '算', '和', '哥', '们', '一', '起', '喝', '酒', '，', '喝', '到', '很', '晚', '，', '也', '要', '提', '前', '打', '电', '话', '告', '诉', '她', '，', '让', '她', '早', '点', '休', '息', '。'], ['我', '是', '一', '爷', '们', '，', '我', '媳', '妇', '绝', '对', '不', '能', '抽', '烟', '，', '喝', '酒', '还', '勉', '强', '过', '得', '去', '，', '不', '过', '该', '喝', '的', '时', '候', '喝', '，', '不', '该', '喝', '的', '时', '候', '，', '少', '扯', '纳', '极', '薄', '蛋', '。'], ['我', '是', '一', '爷', '们', '，', '我', '媳', '妇', '必', '须', '听', '我', '话', '，', '在', '人', '前', '一', '定', '要', '给', '我', '面', '子', '，', '回', '家', '了', '咱', '什', '么', '都', '好', '说', '。'], ['我', '是', '一', '爷', '们', '，', '就', '算', '难', '的', '吃', '不', '上', '饭', '了', '，', '都', '不', '张', '口', '跟', '媳', '妇', '要', '一', '分', '钱', '。'], ['我', '是', '一', '爷', '们', '，', '不', '管', '上', '学', '还', '是', '上', '班', '，', '我', '都', '会', '送', '媳', '妇', '回', '家', '。'], ['我', '是', '一', '爷', '们', '，', '交', '往', '不', '到', '1', '年', '，', '绝', '对', '不', '会', '和', '媳', '妇', '提', '过', '分', '的', '要', '求', '，', '我', '会', '尊', '重', '她', '。'], ['我', '是', '一', '爷', '们', '，', '游', '戏', '永', '远', '比', '不', '上', '我', '媳', '妇', '重', '要', '，', '只', '要', '媳', '妇', '发', '话', '，', '我', '绝', '对', '唯', '命', '是', '从', '。'], ['我', '是', '一', '爷', '们', '，', '上', 'q', '绝', '对', '是', '为', '了', '等', '媳', '妇', '，', '所', '有', '暧', '昧', '的', '心', '情', '只', '为', '她', '一', '个', '女', '人', '而', '写', '，', '我', '不', '一', '定', '会', '经', '常', '写', '日', '志', '，', '可', '是', '我', '会', '告', '诉', '全', '世', '界', '，', '我', '很', '爱', '她', '。'], ['我', '是', '一', '爷', '们', '，', '不', '一', '定', '要', '经', '常', '制', '造', '浪', '漫', '、', '偶', '尔', '过', '个', '节', '日', '也', '要', '送', '束', '玫', '瑰', '花', '给', '媳', '妇', '抱', '回', '家', '。'], ['我', '是', '一', '爷', '们', '，', '手', '机', '会', '24', '小', '时', '为', '她', '开', '机', '，', '让', '她', '半', '夜', '痛', '经', '的', '时', '候', '，', '做', '恶', '梦', '的', '时', '候', '，', '随', '时', '可', '以', '联', '系', '到', '我', '。'], ['我', '是', '一', '爷', '们', '，', '我', '会', '经', '常', '带', '媳', '妇', '出', '去', '玩', '，', '她', '不', '一', '定', '要', '和', '我', '所', '有', '的', '哥', '们', '都', '认', '识', '，', '但', '见', '面', '能', '说', '的', '上', '话', '就', '行', '。'], ['我', '是', '一', '爷', '们', '，', '我', '会', '和', '媳', '妇', '的', '姐', '妹', '哥', '们', '搞', '好', '关', '系', '，', '让', '她', '们', '相', '信', '我', '一', '定', '可', '以', '给', '我', '媳', '妇', '幸', '福', '。'], ['我', '是', '一', '爷', '们', '，', '吵', '架', '后', '、', '也', '要', '主', '动', '打', '电', '话', '关', '心', '她', '，', '咱', '是', '一', '爷', '们', '，', '给', '媳', '妇', '服', '个', '软', '，', '道', '个', '歉', '怎', '么', '了', '？'], ['我', '是', '一', '爷', '们', '，', '绝', '对', '不', '会', '嫌', '弃', '自', '己', '媳', '妇', '，', '拿', '她', '和', '别', '人', '比', '，', '说', '她', '这', '不', '如', '人', '家', '，', '纳', '不', '如', '人', '家', '的', '。'], ['我', '是', '一', '爷', '们', '，', '陪', '媳', '妇', '逛', '街', '时', '，', '碰', '见', '熟', '人', '，', '无', '论', '我', '媳', '妇', '长', '的', '好', '看', '与', '否', '，', '我', '都', '会', '大', '方', '的', '介', '绍', '。'], ['谁', '让', '咱', '爷', '们', '就', '好', '这', '口', '呢', '。'], ['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。'], ['【', '我', '们', '重', '在', '分', '享', '。'], ['所', '有', '文', '字', '和', '美', '图', '，', '来', '自', '网', '络', '，', '晨', '欣', '教', '育', '整', '理', '。'], ['对', '原', '文', '作', '者', '，', '表', '示', '敬', '意', '。'], ['】', '关', '注', '晨', '曦', '教', '育', '[UNK]', '[UNK]', '晨', '曦', '教', '育', '（', '微', '信', '号', '：', 'he', '##bc', '##x', '##jy', '）', '。'], ['打', '开', '微', '信', '，', '扫', '描', '二', '维', '码', '，', '关', '注', '[UNK]', '晨', '曦', '教', '育', '[UNK]', '，', '获', '取', '更', '多', '育', '儿', '资', '源', '。'], ['点', '击', '下', '面', '订', '阅', '按', '钮', '订', '阅', '，', '会', '有', '更', '多', '惊', '喜', '哦', '！']]
    while i < len(document):  # 从文档的第一个位置开始，按个往下看
        segment = document[
            i]  # segment是列表，代表的是按字分开的一个完整句子，如 segment=['我', '是', '一', '爷', '们', '，', '我', '想', '我', '会', '给', '我', '媳', '妇', '最', '好', '的', '幸', '福', '。']
        # segment = get_new_segment(segment)  # whole word mask for chinese: 结合分词的中文的whole mask设置即在需要的地方加上“##”
        current_chunk.append(segment)  # 将一个独立的句子加入到当前的文本块中
        current_length += len(segment)  # 累计到为止位置接触到句子的总长度
        if i == len(document) - 1 or current_length >= target_seq_length:
            # 如果累计的序列长度达到了目标的长度，或当前走到了文档结尾==>构造并添加到“A[SEP]B“中的A和B中；
            if current_chunk:  # 如果当前块不为空
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.

                for m in range(len(current_chunk)):
                    if m == len(current_chunk) - 1:
                        continue
                    tokens_a = []
                    tokens_b = []
                    for j in range(m + 1):
                        tokens_a.extend(current_chunk[j])
                    # Actual next utterance
                    tokens_b.extend(current_chunk[m + 1])

                # a_end = 1
                # if len(current_chunk) >= 2:  # 当前块，如果包含超过两个句子，取当前块的一部分作为“A[SEP]B“中的A部分
                #     a_end = random.randint(1, len(current_chunk) - 1)
                # # 将当前文本段中选取出来的前半部分，赋值给A即tokens_a
                # tokens_a = []
                # for j in range(a_end):
                #     tokens_a.extend(current_chunk[j])
                #
                # # 构造“A[SEP]B“中的B部分(有一部分是正常的当前文档中的后半部;在原BERT的实现中一部分是随机的从另一个文档中选取的，）
                # tokens_b = []
                # for j in range(a_end, len(current_chunk)):
                #     tokens_b.extend(current_chunk[j])

                    # 有百分之50%的概率交换一下tokens_a和tokens_b的位置
                    # print("tokens_a length1:",len(tokens_a))
                    # print("tokens_b length1:",len(tokens_b)) # len(tokens_b) = 0
                    if len(tokens_a) == 0 or len(tokens_b) == 0: continue
                    if random.random() < 0.5:  # 交换一下tokens_a和tokens_b
                        is_random_next = True
                        temp = tokens_a
                        tokens_a = tokens_b
                        tokens_b = temp
                    else:
                        is_random_next = False
                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # 把tokens_a & tokens_b加入到按照bert的风格，即以[CLS]tokens_a[SEP]tokens_b[SEP]的形式，结合到一起，作为最终的tokens; 也带上segment_ids，前面部分segment_ids的值是0，后面部分的值是1.
                    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                    # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                    # They are 1 for the B tokens and the final [SEP]
                    segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                    # 创建masked LM的任务的数据 Creates the predictions for the masked LM objective
                    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                        tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_words)
                    instance = {
                        "tokens": tokens,
                        "segment_ids": segment_ids,
                        "is_random_next": is_random_next,
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_labels": masked_lm_labels}
                    instances.append(instance)
            current_chunk = []  # 清空当前块
            current_length = 0  # 重置当前文本块的长度
        i += 1  # 接着文档中的内容往后看
    return instances


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--max_ngram', default=3, type=int)
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()
    seed_everything(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with args.train_corpus.open() as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    docs.add_document(doc)
                    doc = []
                else:
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
                #debug
                # if len(docs) > 33000:
                #     break
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added

        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        args.output_dir.mkdir(exist_ok=True)
        for epoch in trange(args.epochs_to_generate, desc="Epoch"):
            epoch_filename = args.output_dir / f"epoch_{epoch}.json"
            num_instances = 0
            with epoch_filename.open('w') as epoch_file:
                for doc_idx in trange(len(docs), desc="Document"):
                    doc_instances = create_instances_from_document(
                        docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,max_ngram=args.max_ngram,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        vocab_words=vocab_list)
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            metrics_file = args.output_dir / f"epoch_{epoch}_metrics.json"
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))
    print("finish")


if __name__ == '__main__':
    main()

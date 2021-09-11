from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve

import random
from random import randrange, randint, shuffle, choice, sample
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer
import spacy
import numpy as np
import json
import collections
from svo_extraction import getSVOIDs
nlp = spacy.load("en_core_web_sm")

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
            return self.documents[sampled_doc_index], sampled_doc_index

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


def truncate_seq_pair(tokens_a, tokens_b, max_context_len, max_response_len):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a)
        if total_length <= max_context_len:
            break
        trunc_tokens = tokens_a
        assert len(trunc_tokens) >= 1
        # cut previous tokens for context
        del trunc_tokens[0]

    while True:
        total_length = len(tokens_b)
        if total_length <= max_response_len:
            break
        trunc_tokens = tokens_b
        assert len(trunc_tokens) >= 1
        # cut the last tokens for response
        trunc_tokens.pop()

        # total_length = len(tokens_a) + len(tokens_b)
        # if total_length <= max_num_tokens:
        #     break
        #
        # trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        # assert len(trunc_tokens) >= 1
        #
        # # We want to sometimes truncate from the front and sometimes from the
        # # back to add more randomness and avoid biases.
        # if random() < 0.5:
        #     del trunc_tokens[0]
        # else:
        #     trunc_tokens.pop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, random_shuffle_id):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indexes = []
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
        if (len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
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
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_words)

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels

def create_utterance_mask(tokens_a, tokens_b, masked_lm_prob, max_predictions_per_seq, vocab_list, random_shuffle_id):
    utterance = []
    tokens = []
    masked_lm_positions = []
    masked_lm_labels = []
    u_id = 0
    offset = 0
    eou_ids = []
    for token_item in tokens_a:
        if token_item != "[EOU]":
            utterance.append(token_item)
        else:
            utterance.append("[EOU]")
            eou_idx = offset + len(utterance)
            eou_ids.append(eou_idx) # include an offset of [CLS]
            # if utterance == ['[EOU]']:
            #     print("debug")
            if u_id == random_shuffle_id:
                tokens_i, masked_lm_positions_i, masked_lm_labels_i = create_masked_lm_predictions(
                    utterance, masked_lm_prob, max_predictions_per_seq, vocab_list, random_shuffle_id)
            else:
                tokens_i, masked_lm_positions_i, masked_lm_labels_i = create_masked_lm_predictions(
                    utterance, masked_lm_prob, max_predictions_per_seq, vocab_list, None)
            tokens.extend(tokens_i)
            masked_lm_positions_i = [idx + offset + 1  for idx in masked_lm_positions_i] # add the offset of "[CLS]"
            masked_lm_positions.extend(masked_lm_positions_i)
            masked_lm_labels.extend(masked_lm_labels_i)
            offset += len(utterance)
            utterance = []
            u_id += 1
    tokens_a = tokens
    tokens_b, masked_lm_positions_b, masked_lm_labels_b = create_masked_lm_predictions(
        tokens_b, masked_lm_prob, max_predictions_per_seq, vocab_list, None)
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    offset = len(tokens_a) + 2  # add the offset of "[CLS]" and "[SEP]".
    masked_lm_positions_b = [idx + offset for idx in masked_lm_positions_b]
    masked_lm_positions.extend(masked_lm_positions_b)
    masked_lm_labels.extend(masked_lm_labels_b)
    return tokens, masked_lm_positions, masked_lm_labels, eou_ids

def create_instances_from_document(
        doc_database, doc_idx, svos, max_context_len, max_response_len, max_svo_len, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_list, max_utterance, if_utterance_mask=True):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    document_svo = svos[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_context_len = max_context_len - 2
    max_response_len = max_response_len - 1
    max_seq_length = max_context_len + max_response_len
    max_num_tokens = max_seq_length

    target_seq_length = max_num_tokens
    # if random() < short_seq_prob:
    #     target_seq_length = randint(2, max_num_tokens)

    #  for each dialogue context with multiple utterances we let each utterance (except the first turn)
    #  be positive response of history conversations. And randomly sampled a utterance from the corpus
    #  as the negative example.
    instances = []
    current_chunk = []
    current_svos = []
    current_length = 0
    i = 0

    if len(document) > max_utterance:
        document = document[:max_utterance]
        print("utterance number overlength: ", len(document))

    # utterance order ids
    # shuffle_idx = np.random.permutation(np.arange(len(document)))
    # shuffle_idx = shuffle_idx.tolist()
    # document = [document[i] for i in shuffle_idx]
    # padding = [-100] * (max_utterance - len(shuffle_idx))
    # shuffle_idx += padding

    while i < len(document):
        segment = document[i]
        segment_svo = document_svo[i]
        if len(segment) >= 1:
            current_chunk.append(segment + ["[EOU]"])
            current_length += len(segment) + 1  # add the len of [EOU]
            current_svos.append(segment_svo)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                for m in range(len(current_chunk)):
                    if m == len(current_chunk) - 1:
                        continue

                    if if_utterance_mask:
                        # an utterance that is fully masked
                        random_shuffle_id = random.randint(0, m)
                    else:
                        random_shuffle_id = None

                    # utterance order ids
                    total_chunks = current_chunk[:m + 1]
                    shuffle_idx = np.random.permutation(np.arange(len(total_chunks)))
                    shuffle_idx = shuffle_idx.tolist()
                    total_chunks = [total_chunks[i] for i in shuffle_idx]
                    # padding = [-100] * (max_utterance - len(shuffle_idx))
                    # shuffle_idx += padding

                    tokens_a = []
                    tokens_b = []
                    tokens_a_svo = []
                    tokens_b_svo = []
                    for j in range(m + 1):
                        tokens_a.extend(total_chunks[j])
                        tokens_a_svo.extend(current_svos[j])
                    # Actual next utterance
                    tokens_b.extend(current_chunk[m + 1])
                    if tokens_b[-1] == "[EOU]":
                        tokens_b.pop()
                    tokens_a_2 = tokens_a

                    # Random next utterance
                    # Sample a random document, with longer docs being sampled more frequently
                    random_document, sampled_doc_index = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)
                    random_document = [sent + ["[EOU]"] for sent in random_document if len(sent) >= 1]
                    random_start = randrange(0, len(random_document))
                    tokens_c = random_document[random_start]
                    tokens_c_svo = svos[sampled_doc_index][random_start]

                    truncate_seq_pair(tokens_a, tokens_b, max_context_len, max_response_len)
                    truncate_seq_pair(tokens_a_2, tokens_c, max_context_len, max_response_len)

                    if tokens_a[0] == "[EOU]":
                        del tokens_a[0]
                    if tokens_a_2[0] == "[EOU]":
                        del tokens_a_2[0]
                    
                    assert len(tokens_a) >= 1, print("document:", document)
                    assert len(tokens_b) >= 1, print("current_chunk:", current_chunk)
                    assert len(tokens_c) >= 1, print("random_document", random_document)

                    tokens, masked_lm_positions, masked_lm_labels, eou_ids = create_utterance_mask(tokens_a, tokens_b, masked_lm_prob, max_predictions_per_seq, vocab_list, random_shuffle_id)
                    tokens_neg, masked_lm_positions_neg, masked_lm_labels_neg, eou_ids_neg = create_utterance_mask(tokens_a_2, tokens_c, masked_lm_prob, max_predictions_per_seq, vocab_list, random_shuffle_id)

                    segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
                    segment_ids_neg = [0 for _ in range(len(tokens_a_2) + 2)] + [1 for _ in range(len(tokens_c) + 1)]

                    num_eous = len(eou_ids)
                    # eou_ids=eou_ids[org_nums:]
                    shuffle_idx = shuffle_idx[:num_eous]
                    padding = [-100] * (max_utterance - len(shuffle_idx))
                    shuffle_idx += padding

                    a_svos = getSVOIDs(tokens_a_svo, tokens_a, offset=1)
                    svo_ids = a_svos + getSVOIDs(tokens_b_svo, tokens_b, offset=len(tokens_a) + 2)
                    svo_ids_neg = a_svos + getSVOIDs(tokens_c_svo, tokens_c, offset=len(tokens_a_2) + 2)

                    svo_ids = svo_ids[:max_svo_len]
                    svo_ids_neg = svo_ids_neg[:max_svo_len]

                    instance = {
                        "tokens": tokens,
                        "segment_ids": segment_ids,
                        "is_random_next": False,
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_labels": masked_lm_labels,
                        "position_orders": shuffle_idx,
                        "eou_ids": eou_ids,
                        "svo_ids": svo_ids,
                    }
                    instances.append(instance)

                    instance_neg = {
                        "tokens": tokens_neg,
                        "segment_ids": segment_ids_neg,
                        "is_random_next": True,
                        "masked_lm_positions": masked_lm_positions_neg,
                        "masked_lm_labels": masked_lm_labels_neg,
                        "position_orders": shuffle_idx,
                        "eou_ids": eou_ids,
                        "svo_ids": svo_ids_neg,
                    }
                    instances.append(instance_neg)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument('--svo_corpus', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--if_utterance_mask", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_context_len", type=int, default=448)
    parser.add_argument("--max_response_len", type=int, default=64)
    parser.add_argument("--max_utterance", type=int, default=20)
    parser.add_argument("--max_svo_len", type=int, default=10)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # add special tokens
    print(tokenizer.additional_special_tokens)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[EOU]']})
    print(tokenizer.additional_special_tokens)
    print(tokenizer.additional_special_tokens_ids)

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
                # debug
                # if len(docs) > 33000:
                #     break
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
        svos = []
        with args.svo_corpus.open() as f:
            svo = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    svos.append(svo)
                    svo = []
                else:
                    line = line.strip()
                    svo_item = eval(line)
                    svo.append(eval(line))
            if svo:
                svos.append(svo)  # If the last doc didn't end on a newline, make sure it still gets added

        assert len(svos) == len(docs)

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
                        docs, doc_idx, svos, max_context_len=args.max_context_len, max_response_len=args.max_response_len,
                        max_svo_len = args.max_svo_len,
                        short_seq_prob=args.short_seq_prob,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        vocab_list=vocab_list, max_utterance=args.max_utterance, if_utterance_mask=args.if_utterance_mask)
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            metrics_file = args.output_dir / f"epoch_{epoch}_metrics.json"
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_context_len + args.max_response_len,
                    "max_utterance": args.max_utterance,
                    "max_svo_len": args.max_svo_len
                }
                metrics_file.write(json.dumps(metrics))
    print("finish")


if __name__ == '__main__':
    main()

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
import spacy
from random import random, randrange, randint, shuffle, choice, sample
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import json
from svo_extraction import extractSVOs
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
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, max_svo_len, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    target_seq_length = max_num_tokens
    # if random() < short_seq_prob:
    #     target_seq_length = randint(2, max_num_tokens)

    #  for each dialogue context with multiple utterances we let each utterance (except the first turn)
    #  be positive response of history conversations. And randomly sampled a utterance from the corpus
    #  as the negative example.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        pew_svos = []
        pre_len = 0
        if len(segment)>=1:
            current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                for m in range(len(current_chunk)):
                    if m == len(current_chunk) - 1:
                        continue
                    tokens_a = []
                    tokens_b = []
                    for j in range(m + 1):
                        tokens_a.extend(current_chunk[j])
                    # Actual next utterance
                    tokens_b.extend(current_chunk[m + 1])

                    tokens_a_2 = tokens_a
                    # Random next utterance
                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)
                    random_document = [sent for sent in random_document if len(sent)>=1]
                    random_start = randrange(0, len(random_document))
                    tokens_c = random_document[random_start]

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    truncate_seq_pair(tokens_a_2, tokens_c, max_num_tokens)

                    assert len(tokens_a) >= 1 , print("document:",document)
                    assert len(tokens_b) >= 1, print("current_chunk:",current_chunk)
                    assert len(tokens_c) >= 1 , print("random_document",random_document)

                    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                    tokens_neg = ["[CLS]"] + tokens_a_2 + ["[SEP]"] + tokens_c + ["[SEP]"]
                    # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                    # They are 1 for the B tokens and the final [SEP]
                    segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
                    segment_ids_neg = [0 for _ in range(len(tokens_a_2) + 2)] + [1 for _ in range(len(tokens_c) + 1)]

                    # if m > 0 and len(tokens_a) > pre_len:
                    #     new_toks = tokens_a[pre_len:]
                    #     a_svos = pew_svos + extractSVOs(nlp, new_toks, offset=pre_len)
                    #     pew_svos = a_svos
                    # else:
                    #     a_svos = extractSVOs(nlp, tokens_a, offset=1)
                    #     pre_len = len(tokens_a) + 1
                    #     pew_svos = a_svos

                    # svo_ids = [a_svos + extractSVOs(nlp, tokens_b, offset=len(tokens_a)+2)]
                    # svo_ids_neg = [a_svos + extractSVOs(nlp, tokens_c, offset=len(tokens_a_2)+2)]
                    a_svos = extractSVOs(nlp, tokens_a, offset=1)
                    svo_ids = a_svos + extractSVOs(nlp, tokens_b, offset=len(tokens_a) + 2)
                    svo_ids_neg = a_svos + extractSVOs(nlp, tokens_c, offset=len(tokens_a_2) + 2)

                    svo_ids = svo_ids[:max_svo_len]
                    svo_ids_neg = svo_ids_neg[:max_svo_len]

                    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)
                    tokens_neg, masked_lm_positions_neg, masked_lm_labels_neg = create_masked_lm_predictions(
                        tokens_neg, masked_lm_prob, max_predictions_per_seq, vocab_list)

                    instance = {
                        "tokens": tokens,
                        "segment_ids": segment_ids,
                        "is_random_next": False,
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_labels": masked_lm_labels,
                        "svo_ids": svo_ids}
                    instances.append(instance)

                    instance_neg = {
                        "tokens": tokens_neg,
                        "segment_ids": segment_ids_neg,
                        "is_random_next": True,
                        "masked_lm_positions": masked_lm_positions_neg,
                        "masked_lm_labels": masked_lm_labels_neg,
                        "svo_ids": svo_ids_neg}
                    instances.append(instance_neg)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")

    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--max_svo_len", type=int, default=10)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()

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
                        docs, doc_idx, max_seq_length=args.max_seq_len, max_svo_len=args.max_svo_len, short_seq_prob=args.short_seq_prob,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        vocab_list=vocab_list)
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            metrics_file = args.output_dir / f"epoch_{epoch}_metrics.json"
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len,
                    "max_svo_len": args.max_svo_len
                }
                metrics_file.write(json.dumps(metrics))
    print("finish")


if __name__ == '__main__':
    main()

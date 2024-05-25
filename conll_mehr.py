import glob
import torch
from torchtext.vocab import Vectors
import random
import os
import io
import re
from utils import flatten
from cached_property import cached_property
from copy import deepcopy as c
from boltons.iterutils import pairwise


class LazyVectors:
    """Load only those vectors from GloVE that are in the vocab.
    Assumes PAD id of 0 and UNK id of 1
    """

    unk_idx = 1

    def __init__(self, name,
                 cache,
                 skim=None,
                 vocab=None):
        """  Requires the glove vectors to be in a folder named .vector_cache
        Setup:
            >> cd ~/where_you_want_to_save
            >> mkdir .vector_cache
            >> mv ~/where_glove_vectors_are_stored/glove.840B.300d.txt
                ~/where_you_want_to_save/.vector_cache/glove.840B.300d.txt
        Initialization (first init will be slow):
            >> VECTORS = LazyVectors(cache='~/where_you_saved_to/.vector_cache/',
                                     vocab_file='../path/vocabulary.txt',
                                     skim=None)
        Usage:
            >> weights = VECTORS.weights()
            >> embeddings = torch.nn.Embedding(weights.shape[0],
                                              weights.shape[1],
                                              padding_idx=0)
            >> embeddings.weight.data.copy_(weights)
            >> embeddings(sent_to_tensor('kids love unknown_word food'))
        You can access these moved vectors from any repository
        """
        self.__dict__.update(locals())
        if self.vocab is not None:
            self.set_vocab(vocab)

    @classmethod
    def from_corpus(cls, corpus_vocabulary, name, cache):
        return cls(name=name, cache=cache, vocab=corpus_vocabulary)

    @cached_property
    def loader(self):
        return Vectors(self.name, cache=self.cache)

    def set_vocab(self, vocab):
        """ Set corpus vocab
        """
        # Intersects and initializes the torchtext Vectors class
        self.vocab = [v for v in vocab if v in self.loader.stoi][:self.skim]

        self.set_dicts()

    def get_vocab(self, filename):
        """ Read in vocabulary (top 30K words, covers ~93.5% of all tokens) """
        return read_corpus(filename)

    def set_dicts(self):
        """ _stoi: map string > index
            _itos: map index > string
        """
        self._stoi = {s: i for i, s in enumerate(self.vocab)}
        self._itos = {i: s for s, i in self._stoi.items()}

    def weights(self):
        """Build weights tensor for embedding layer """
        # Select vectors for vocab words.
        weights = torch.stack([
            self.loader.vectors[self.loader.stoi[s]]
            for s in self.vocab
        ])

        # Padding + UNK zeros rows.
        return torch.cat([
            torch.zeros((2, self.loader.dim)),
            weights,
        ])

    def stoi(self, s):
        """ String to index (s to i) for embedding lookup """
        idx = self._stoi.get(s)
        return idx + 2 if idx else self.unk_idx

    def itos(self, i):
        """ Index to string (i to s) for embedding lookup """
        token = self._itos.get(i)
        return token if token else 'UNK'


class Corpus:
    def __init__(self, documents):
        self.docs = documents
        self.vocab, self.char_vocab = self.get_vocab()

    def __getitem__(self, idx):
        return self.docs[idx]

    def __repr__(self):
        return 'Corpus containg %d documents' % len(self.docs)

    def get_vocab(self):
        """ Set vocabulary for LazyVectors """
        vocab, char_vocab = set(), set()
        for document in self.docs:
            vocab.update(document.tokens)
            char_vocab.update([char
                               for word in document.tokens
                               for char in word])

        return vocab, char_vocab


class Document:
    def __init__(self, raw_text, tokens, corefs, filename):
        self.raw_text = raw_text
        self.tokens = tokens
        self.corefs = corefs
        self.filename = filename

        # Filled in at evaluation time.
        self.tags = None

    def __getitem__(self, idx):
        return (self.tokens[idx], self.corefs[idx],
                )

    def __repr__(self):
        return 'Document containing %d tokens' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    @cached_property
    def sents(self):
        """ Regroup raw_text into sentences """

        # Get sentence boundaries while avoiding periods within numbers
        sent_idx = []
        for idx, token in enumerate(self.tokens):
            if token in ['.', '?', '!']:
                # Check if the token is a period not surrounded by digits
                if not (token == '.' and 0 < idx < len(self.tokens) - 1 and self.tokens[idx - 1].isdigit() and
                        self.tokens[idx + 1].isdigit()):
                    sent_idx.append(idx + 1)

        # Regroup (returns list of lists)
        return [self.tokens[i1:i2] for i1, i2 in pairwise([0] + sent_idx)]

    def truncate(self, MAX=50):
        """ Randomly truncate the document to up to MAX sentences """
        if len(self.sents) > MAX:
            i = random.sample(range(MAX, len(self.sents)), 1)[0]
            tokens = flatten(self.sents[i - MAX:i])
            return self.__class__(c(self.raw_text), tokens,
                                  c(self.corefs), c(self.speakers),
                                  c(self.genre), c(self.filename))
        return self


def load_file(filename):
    """ Load a *._conll file
    Input:
        filename: path to the file
         Output:
        documents: list of Document class for each document in the file containing:
         tokens:                   split list of text
         utts_corefs:
                coref['label']:     id of the coreference cluster
                coref['start']:     start index (index of first token in the utterance)
                coref['end':        end index (index of last token in the utterance)
                coref['span']:      corresponding span
    """
    documents = []
    print(filename)
    with io.open(filename, 'rt', encoding='utf-8', errors='strict') as f:
        raw_text, tokens, text, utts_corefs, corefs, index = [], [], [], [], [], 0
        for line in f:
            raw_text.append(line)
            cols = line.split()
            try:
                # End of sentence within a document:
                if len(cols) == 0:
                    if text:
                        tokens.extend(text), utts_corefs.extend(corefs)
                        text, corefs = [], []
                        continue
                # End of document: organize the data, append to output, reset variables for next document.
                elif len(cols) == 2:
                    doc = Document(raw_text, tokens, utts_corefs, filename)
                    documents.append(doc)
                    raw_text, tokens, text, utts_corefs, index = [], [], [], [], 0
                elif len(cols) > 7:
                    text.append(cols[3])  # add token
                    # If the last column isn't a '-', there is a coreference link
                    if cols[-1] != u'-':
                        coref_expr = cols[-1].split(u'|')
                        for token in coref_expr:
                            # Check if coref column token entry contains (, a number, or ).
                            match = re.match(r"^(\(?)(\d+)(\)?)$", token)
                            label = match.group(2)

                            # If it does, extract the coref label, its start index,
                            if match.group(1) == u'(':
                                corefs.append({'label': label,
                                               'start': index,
                                               'end': None})
                            if match.group(3) == u')':
                                for i in range(len(corefs) - 1, -1, -1):
                                    if corefs[i]['label'] == label and corefs[i]['end'] is None:
                                        break
                                # Extract the end index, include start and end indexes in 'span'
                                corefs[i].update({'end': index,
                                                  'span': (corefs[i]['start'], index)})
                    index += 1
                else:
                    continue
            except:
                index += 1
                print("Error occurred while processing line:")
    return documents


def read_corpus(path):
    """
    read all files in current directory.
    :param path: the path of corpus
    :return: all structured files in corpus
    """
    conll_files = glob.glob(os.path.join(path, '*.conll'))
    return Corpus(flatten([load_file(file) for file in conll_files]))


GLOVE = LazyVectors.from_corpus(read_corpus('data/Mehr/train-dev/').vocab,
                                name='glove_arman_300.txt',
                                cache='data/vectors/')

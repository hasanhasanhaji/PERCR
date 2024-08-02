import glob
import torch
import torchtext
import random
import os
import attr
import io
import re
from utils import flatten
from cached_property import cached_property
from copy import deepcopy as c
from boltons.iterutils import pairwise
import configparser

torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import Vectors

config = configparser.ConfigParser()
config.read('config.ini')  # Load configuration from file


@attr.s(frozen=True, repr=False)
class Span:
    # Left / right token indexes
    i1 = attr.ib()
    i2 = attr.ib()

    # Id within total spans (for indexing into a batch computation)
    id = attr.ib()

    # # Speaker
    # speaker = attr.ib()
    #
    # # Genre
    # genre = attr.ib()

    # Unary mention score, as tensor
    si = attr.ib(default=None)

    # List of candidate antecedent spans
    yi = attr.ib(default=None)

    # Corresponding span ids to each yi
    yi_idx = attr.ib(default=None)

    def __len__(self):
        return self.i2 - self.i1 + 1

    def __repr__(self):
        return 'Span representing %d tokens' % (self.__len__())


class LazyVectors:
    """Load only those vectors from GloVE that are in the vocab.
    Assumes PAD id of 0 and UNK id of 1
    name: The name of the embedding file (e.g., 'glove.840B.300d.txt').
    cache: The directory where the embedding file is stored.
    skim: An optional limit on the number of words to load from the embeddings.
    vocab: An optional list of words in the vocabulary.
    """

    unk_idx = 1  # Index used for unknown words, set to 1.

    def __init__(self, name,
                 cache,
                 skim=None,
                 vocab=None):
        """  The LazyVectors class is designed to load pre-trained word vectors
        (like GloVe) efficiently by loading only
        those vectors that are in the specified vocabulary.
        """
        self.__dict__.update(locals())
        if self.vocab is not None:
            self.set_vocab(vocab)

    @classmethod
    def from_corpus(cls, corpus_vocabulary, name, cache):
        """
        from_corpus: A class method to create an instance
        using a vocabulary from a corpus, embedding name, and cache directory.
        """
        return cls(name=name, cache=cache, vocab=corpus_vocabulary)

    @cached_property
    def loader(self):
        """
        loader: A cached property that initializes and returns
        a Vectors object (likely from torchtext) which loads the embedding vectors.
        :return:
        """
        return Vectors(self.name, cache=self.cache)

    def set_vocab(self, vocab):
        """
        set_vocab: Filters the provided vocabulary to include only those words present
        in the embeddings and initializes the dictionary mappings.
        """
        # Intersects and initializes the torchtext Vectors class
        self.vocab = [v for v in vocab if v in self.loader.stoi][:self.skim]

        self.set_dicts()

    def get_vocab(self, filename):
        """ Read in vocabulary (top 30K words, covers ~93.5% of all tokens) """
        return read_corpus(filename)

    def set_dicts(self):
        """
        set_dicts: Sets up the string-to-index (_stoi) and
        index-to-string (_itos) mappings for the vocabulary.
        """
        self._stoi = {s: i for i, s in enumerate(self.vocab)}
        self._itos = {i: s for s, i in self._stoi.items()}

    def weights(self):
        """
        Build weights tensor for embedding layer
        """
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
        """
        Get the vocab and char vocab of documents.
        :return:
        vocab: A set of unique tokens (words) found across all documents in the corpus.
        char_vocab: A set of unique characters found across all tokens.
        """
        vocab, char_vocab = set(), set()
        for document in self.docs:
            vocab.update(document.tokens)
            char_vocab.update([char
                               for word in document.tokens
                               for char in word])
        return vocab, char_vocab

    def split_corpus(self, dev_size=None, seed=42):
        """ Split the corpus into training and development sets """
        if dev_size is None:  # Check if dev_size was provided
            # Load dev_size from configuration
            dev_size = config.getfloat('TRAINING', 'dev_size')
        random.seed(seed)
        random.shuffle(self.docs)
        split_idx = int(len(self.docs) * (1 - dev_size))
        train_docs = self.docs[:split_idx]
        dev_docs = self.docs[split_idx:]
        return Corpus(train_docs), Corpus(dev_docs)


class Document:
    """
    The class Document for preprocessing documents.
    """

    def __init__(self, raw_text, tokens, corefs, filename):
        self.raw_text = raw_text
        self.tokens = tokens
        self.corefs = corefs
        self.filename = filename

        # Filled in at evaluation time.
        self.tags = None

    def __getitem__(self, idx):
        """
            It means you can access tokens and coreferences of a document like this:
            doc = Document(...)  # Create a Document object
            token, coref = doc[5]  # Get the token and coreference at index 5
        """
        return self.tokens[idx], self.corefs[idx]

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
            if token in ['.']:
                # Check if the token is a period not surrounded by digits
                if not (token == '.' and 0 < idx < len(self.tokens) - 1 and self.tokens[idx - 1].isdigit() and
                        self.tokens[idx + 1].isdigit()):
                    sent_idx.append(idx + 1)

        # Regroup (returns list of lists)
        return [self.tokens[i1:i2] for i1, i2 in pairwise([0] + sent_idx)]

    def truncate(self, MAX=config.getint('TRAINING', 'max_sentences_per_doc')):
        """ Randomly truncate the document to up to MAX sentences """
        if len(self.sents) > MAX:
            i = random.sample(range(MAX, len(self.sents)), 1)[0]
            tokens = flatten(self.sents[i - MAX:i])
            return self.__class__(c(self.raw_text), tokens,
                                  c(self.corefs), c(self.speakers),
                                  c(self.genre), c(self.filename))
        return self


def load_mehr_file(filename):
    """
     The function processes the CoNLL file and extracts relevant data for coreference resolution,
     organizing it into a list of Document objects.
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

    with io.open(filename, 'rt', encoding='utf-8', errors='strict') as f:
        # index: Token index within the document.
        # corefs: List of ongoing coreference information of current sentence.
        # utts_corefs: List of coreference information for the current document.
        # tokens: List of individual tokens of current document.
        # text : List of individual tokens of current sentence.
        raw_text, tokens, text, utts_corefs, corefs, index = [], [], [], [], [], 0
        for line in f:
            raw_text.append(line)
            cols = line.split()
            try:
                # End of sentence within a document for MEHR corpus.
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
                # If the line has more than seven columns, it's assumed to be a token line.
                elif len(cols) > 7:
                    text.append(cols[3])  # add token to current line tokens
                    # If the last column isn't a '-', there is a coreference link
                    if cols[-1] != u'-':
                        coref_expr = cols[-1].split(u'|')
                        for token in coref_expr:
                            # Check if coref column token entry contains (, a number, or ).
                            match = re.match(r"^(\(?)(\d+)(\)?)$", token)
                            label = match.group(2)

                            # If it does, extract the coref label, its start index,
                            if match.group(1) == u'(':  # start of coref expression
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


            except Exception as e:
                index += 1
                print(filename)
                print(f"Error processing line: {line}")  # Print the line causing the error

                continue  # Continue to the next line after logging the error
                # print("Error occurred while processing line:")

    return documents


def load_rcdat_file(filename):
    documents = []
    current_sentence_index = 0
    with io.open(filename, 'rt', encoding='utf-8', errors='strict') as f:
        raw_text, tokens, utts_corefs, corefs = [], [], [], []
        current_chain = None

        for line in f:
            raw_text.append(line)
            cols = line.split()
            if not cols:
                continue

            sentence_index = int(cols[1])
            if sentence_index != current_sentence_index:
                utts_corefs.extend(corefs)
                corefs = []
                current_sentence_index = sentence_index
                current_chain = None

            token = cols[3]
            tokens.append(token)
            coref_column = cols[-1]

            if coref_column != "-":
                label = ""
                i = 0

                while i < len(coref_column):
                    char = coref_column[i]
                    if char.isdigit():
                        label += char
                    elif char == '(':
                        if current_chain is None or current_chain['label'] != label:
                            current_chain = {'label': label, 'start': len(tokens) - 1, 'end': None}
                            corefs.append(current_chain)
                        label = ""
                    elif char == ')':
                        if current_chain is not None and current_chain['label'] == label:
                            current_chain['end'] = len(tokens) - 1
                            current_chain['span'] = (current_chain['start'], current_chain['end'])  # Update span here
                            current_chain = None
                        label = ""
                    elif char == '*':
                        pass

                    i += 1

        if tokens:
            tokens.append('.')  # Add period to the last sentence
        utts_corefs.extend(corefs)  # Add corefs from last sentence to the document
        doc = Document(raw_text, tokens, utts_corefs, filename)
        documents.append(doc)

    return documents



def read_corpus(path, corpus_type):
    """
    Reads all files in the current directory based on the specified corpus type.

    Args:
        path (str): The path to the corpus directory.
        corpus_type (str): The type of corpus ("mehr" or "rcdat").

    Returns:
        Corpus: A `Corpus` object containing the loaded documents.
    """
    load_function = load_mehr_file if corpus_type == "Mehr" else load_rcdat_file
    conll_files = glob.glob(os.path.join(path, '*.conll'))
    return Corpus(flatten([load_function(file) for file in conll_files]))


def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def lookup_tensor(tokens, vectorizer):
    """ Convert a sentence to an embedding lookup tensor """
    return to_cuda(torch.tensor([vectorizer.stoi(t) for t in tokens]))

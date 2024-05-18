import glob
import os
import io
import re
from utils import flatten


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
                self.speakers[idx], self.genre)

    def __repr__(self):
        return 'Document containing %d tokens' % len(self.tokens)

    def __len__(self):
        return len(self.tokens)


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

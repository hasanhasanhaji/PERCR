import torch
from collections import defaultdict
from itertools import groupby, combinations

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]


def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def extract_gold_corefs(document):
    """ Parse coreference dictionary of a document to get coref links """

    # Initialize defaultdict for keeping track of corefs
    gold_links = defaultdict(list)

    # Compute number of mentions
    gold_mentions = set([coref['span'] for coref in document.corefs])
    total_mentions = len(gold_mentions)

    # Compute number of coreferences
    for coref_entry in document.corefs:
        # Parse label of coref span, the span itself
        label, span_idx = coref_entry['label'], coref_entry['span']

        # All spans corresponding to the same label
        gold_links[label].append(span_idx)  # get all spans corresponding to some label

    # Flatten all possible corefs, sort, get number
    gold_corefs = flatten([[coref
                            for coref in combinations(gold, 2)]
                           for gold in gold_links.values()])
    gold_corefs = sorted(gold_corefs)
    total_corefs = len(gold_corefs)

    return gold_corefs, total_corefs, gold_mentions, total_mentions

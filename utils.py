import torch
from collections import defaultdict
from itertools import groupby, combinations
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence

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


def pack(tensors):
    """ Pack list of tensors, provide reorder indexes """

    # Get sizes
    sizes = [t.shape[0] for t in tensors]

    # Get indexes for sorted sizes (largest to smallest)
    size_sort = np.argsort(sizes)[::-1]

    # Resort the tensor accordingly
    sorted_tensors = [tensors[i] for i in size_sort]

    # Resort sizes in descending order
    sizes = sorted(sizes, reverse=True)

    # Pack the padded sequences
    packed = pack_sequence(sorted_tensors)

    # Regroup indexes for restoring tensor to its original order
    reorder = torch.tensor(np.argsort(size_sort), requires_grad=False)

    return packed, reorder

def unpack_and_unpad(lstm_out, reorder):
    """ Given a padded and packed sequence and its reordering indexes,
    unpack and unpad it. Inverse of pad_and_pack """

    # Restore a packed sequence to its padded version
    unpacked, sizes = pad_packed_sequence(lstm_out, batch_first=True)

    # Restored a packed sequence to its original, unequal sized tensors
    unpadded = [unpacked[idx][:val] for idx, val in enumerate(sizes)]

    # Restore original ordering
    regrouped = [unpadded[idx] for idx in reorder]

    return regrouped
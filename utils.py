import torch


def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]


def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

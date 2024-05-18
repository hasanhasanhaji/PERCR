import glob
import os
from utils import *


def load_file(filename):
    pass


def read_corpus(path):
    """
        read all files in current directory.
        :param path: the path of corpus
        :return: all structured files in corpus
        """
    conll_files = glob.glob(os.path.join(path, '*.conll'))
    x = flatten([load_file(file) for file in conll_files])
    pass
    return flatten([load_file(file) for file in conll_files])


read_corpus('data/Mehr/train-dev/')

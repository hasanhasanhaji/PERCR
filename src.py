import logging

import torch.nn as nn
import torch.nn.functional as F

from conll_mehr import *
from utils import *

# configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class CharCNN(nn.Module):
    """ Character-level CNN. Contains character embeddings.
    Give a sentence then return character embeddings.
    """

    unk_idx = 1
    vocab = read_corpus('data/Mehr/train-dev').char_vocab
    _stoi = {char: idx + 2 for idx, char in enumerate(vocab)}
    pad_size = 15

    def __init__(self, filters, char_dim=8):
        super().__init__()

        self.embeddings = nn.Embedding(len(self.vocab) + 2, char_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.pad_size,
                                              out_channels=filters,
                                              kernel_size=n) for n in (3, 4, 5)])

    def forward(self, sent):
        """ Compute filter-dimensional character-level features for each doc token """
        embedded = self.embeddings(self.sent_to_tensor(sent))
        convolved = torch.cat([F.relu(conv(embedded)) for conv in self.convs], dim=2)
        pooled = F.max_pool1d(convolved, convolved.shape[2]).squeeze(2)
        return pooled

    def sent_to_tensor(self, sent):
        """ Batch-ify a document class instance for CharCNN embeddings """
        tokens = [self.token_to_idx(t) for t in sent]
        batch = self.char_pad_and_stack(tokens)
        return batch

    def token_to_idx(self, token):
        """ Convert a token to its character lookup ids """
        return to_cuda(torch.tensor([self.stoi(c) for c in token]))

    def char_pad_and_stack(self, tokens):
        """ Pad and stack an uneven tensor of token lookup ids """
        skimmed = [t[:self.pad_size] for t in tokens]

        lens = [len(t) for t in skimmed]

        padded = [F.pad(t, (0, self.pad_size - length))
                  for t, length in zip(skimmed, lens)]

        return torch.stack(padded)

    def stoi(self, char):
        """ Lookup char id. <PAD> is 0, <UNK> is 1. """
        idx = self._stoi.get(char)
        return idx if idx else self.unk_idx


class DocumentEncoder(nn.Module):
    """ Document encoder for tokens The class takes hidden_dim (dimensionality of hidden states), char_filters (
    number of filters in the character-level CNN), and n_layers (number of layers in the LSTM) as parameters.
    """

    def __init__(self, hidden_dim, char_filters, n_layers=2):
        super().__init__()

        #  Unit vector embeddings >>> normalization
        logger.info("Start normalizing glove weights.")
        glove_weights = F.normalize(GLOVE.weights())  # unique vocabs ** 300 (glove dim)

        # GLoVE
        self.glove = nn.Embedding(glove_weights.shape[0], glove_weights.shape[1])
        self.glove.weight.data.copy_(glove_weights)
        self.glove.weight.requires_grad = False

        # Character embedding
        self.char_embeddings = CharCNN(char_filters)  # Create char nn layer for a sentence

        # Sentence-LSTM
        self.lstm = nn.LSTM(glove_weights.shape[1] + char_filters,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout
        self.emb_dropout = nn.Dropout(0.50, inplace=True)
        self.lstm_dropout = nn.Dropout(0.20, inplace=True)



class CorefModel(nn.Module):
    """
    Coreference resolution model. This class handles encoder and scoring links.
    """

    def __init__(self, embed_dim, hidden_dim, encoder_type, char_filters=50, distance_dim=20):
        super().__init__()

        # Define base hyperparameters (applicable to all encoders)
        self.distance_dim = distance_dim

        # Handle encoder-specific hyperparameters and initialization
        if encoder_type == "lstm":
            # Forward and backward passes, avg'd attn over embeddings, span width
            self.gi_dim = embed_dim * 3 + self.distance_dim

            # gi, gj, gi*gj, distance between gi and gj
            self.gij_dim = self.gi_dim * 3 + self.distance_dim

            logger.info(f"For Bi-LSTM encoder: span_dim is {self.gi_dim}, pairs_dim is {self.gij_dim}")
            self.encoder = DocumentEncoder(hidden_dim, char_filters)
            # self.score_spans = MentionScore(self.gi_dim, embed_dim, self.distance_dim)
            # self.score_pairs = PairwiseScore(self.gij_dim, distance_dim)
        else:
            try:
                pass
                # # Import and initialize PARSBERT encoder (or any compatible BERT model)
                # self.encoder = AutoModel.from_pretrained(encoder_type)
                # self.tokenizer = AutoTokenizer.from_pretrained(encoder_type)
                #
                # # Extract embedding dimension from BERT model (assuming last hidden layer)
                # self.embed_dim = self.encoder.config.hidden_size
                #
                # # Calculate gi_dim and gij_dim based on BERT embedding dimension
                # self.gi_dim = self.embed_dim * 3 + distance_dim
                # self.gij_dim = self.gi_dim * 3 + distance_dim
                # logger.info( f"For BERT encoder (hidden_dim={self.embed_dim}): span_dim is {self.gi_dim},"
                # f" pairs_dim is {self.gij_dim}")
                # self.encoder = BERTEncoder(encoder_type)
                # self.score_spans = MentionScore(self.gi_dim, self.embed_dim, self.distance_dim)
                # self.score_pairs = PairwiseScore(self.gij_dim, distance_dim)
            except ImportError:
                logging.warning("transformers library not found. Using default hyperparameters.")


if __name__ == "__main__":
    # Create coreference resolution model

    # embeds_dim = the dimensionality of token embeddings
    # hidden dim = the hidden dim of LSTM

    # Encoder type = can choose between ['lstm','HooshvareLab/bert-fa-zwnj-base']
    logger.info("Creating coref model...")
    model = CorefModel(embed_dim=400, hidden_dim=200, encoder_type='lstm')

    logger.info("Reading training and test corpora...")

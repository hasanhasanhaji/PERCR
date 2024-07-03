import logging
import random
import os
import re
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from conll_mehr import *
from utils import *
from datetime import datetime

# configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class CharCNN(nn.Module):
    """ Character-level CNN. Contains character embeddings.
    Give a sentence then return character embeddings.
    """

    unk_idx = 1  # Sets the index for unknown characters.
    vocab = read_corpus('data/Mehr/train-dev').char_vocab  # Loads the character vocabulary from the training data
    _stoi = {char: idx + 2 for idx, char in enumerate(
        vocab)}  # Creates a dictionary mapping each character to an index, starting from 2 to reserve indices 0 and
    # 1 for padding and unknown characters.
    pad_size = 15  # Sets the fixed size for padding sequences.

    def __init__(self, filters, char_dim=8):
        super().__init__()

        self.embeddings = nn.Embedding(len(self.vocab) + 2, char_dim,
                                       padding_idx=0)  # Creates an embedding layer for character embeddings,
        # with padding index set to 0.
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.pad_size,
                                              out_channels=filters,
                                              kernel_size=n) for n in (3, 4, 5)])
        # Creates a list of convolutional layers with different kernel sizes (3, 4, and 5).

    def forward(self, sent):
        """
         Compute filter-dimensional character-level features for each doc token
         """
        # TODO:
        pass

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
        word2vec_weights = F.normalize(W2VEC.weights())

        # GLoVE
        self.glove = nn.Embedding(glove_weights.shape[0], glove_weights.shape[1])
        self.glove.weight.data.copy_(glove_weights)  # initializes the embedding layer
        # with the pre-trained vectors instead of random weights.
        self.glove.weight.requires_grad = False

        self.word2vec = nn.Embedding(word2vec_weights.shape[0], word2vec_weights.shape[1])
        self.word2vec.weight.data.copy_(word2vec_weights)
        self.word2vec.weight.requires_grad = False

        # Character embedding
        self.char_embeddings = CharCNN(char_filters)  # Create char nn layer for a sentence

        # Sentence-LSTM
        self.lstm = nn.LSTM(glove_weights.shape[1] + word2vec_weights.shape[1] + char_filters,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout
        self.emb_dropout = nn.Dropout(0.50, inplace=True)
        self.lstm_dropout = nn.Dropout(0.20, inplace=True)  # Applied to the outputs of the LSTM layers.

    def forward(self, doc):
        """
          Convert document words to ids, embed them, pass through LSTM.
        :param doc:
        :return:
        """
        # Embed document
        embeds = [self.embed(s) for s in doc.sents]

        # Batch for LSTM
        packed, reorder = pack(embeds)

        # Apply embedding dropout
        self.emb_dropout(packed[0])

        # Pass an LSTM over the embeds
        output, _ = self.lstm(packed)

        # Apply dropout
        self.lstm_dropout(output[0])

        # Undo the packing/padding required for batching
        states = unpack_and_unpad(output, reorder)

        return torch.cat(states, dim=0), torch.cat(embeds, dim=0)


    def embed(self, sent):
        """ Embed a sentence using GLoVE, word2vec, and character embeddings """

        # Embed the tokens with Glove
        glove_embeds = self.glove(lookup_tensor(sent, GLOVE))

        # Embed again using Turian this time
        word2vec_embeds = self.word2vec(lookup_tensor(sent, W2VEC))

        # Character embeddings
        char_embeds = self.char_embeddings(sent)

        # Concatenate them all together
        embeds = torch.cat((glove_embeds, word2vec_embeds, char_embeds), dim=1)

        return embeds


class Score(nn.Module):
    """
    The Score class is a generic scoring module designed to
    process input features and output a scalar score.
     It uses a series of fully connected (linear) layers,
     ReLU activations, and dropout layers to perform this task.
    """

    def __init__(self, embeds_dim, hidden_dim=150):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)


class Distance(nn.Module):
    """ Learned, continuous representations for: span widths, distance
    between spans
    """

    bins = [1, 2, 3, 4, 8, 16, 32, 64]

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins) + 1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return to_cuda(torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        ))


class MentionScore(nn.Module):
    """
    Mention scoring module
    """

    def __init__(self, gi_dim, attn_dim, distance_dim):
        super().__init__()

        self.attention = Score(attn_dim)  # Computes attention scores for the spans.
        self.width = Distance(distance_dim)  # processes distance-related features.
        self.score = Score(gi_dim)  # Computes the final mention scores using combined features.

    def forward(self, states, embeds, doc, K=250):
        """ Compute unary mention score for each span
        """
        # TODO:
        pass


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module
    """

    def __init__(self, gij_dim, distance_dim):
        super().__init__()

        self.distance = Distance(distance_dim)
        self.score = Score(gij_dim)

    def forward(self, spans, g_i, mention_scores):
        """ Compute pairwise score for spans and their up to K antecedents
        """
        pass
        # TODO:


class CorefModel(nn.Module):
    """
    Coreference resolution model. This class handles encoder and scoring links.
    It computes coreference links between spans.
    """

    def __init__(self, embed_dim, hidden_dim, encoder_type, char_filters=50, distance_dim=20):
        super().__init__()

        # Define base hyperparameters (applicable to all encoders)
        self.distance_dim = distance_dim

        # Handle encoder-specific hyperparameters and initialization
        if encoder_type == "lstm":

            # Forward and backward pass of bi-lstm over the document
            attn_dim = hidden_dim * 2

            # Forward and backward passes, avg'd attn over embeddings, span width
            self.gi_dim = attn_dim * 2 + embed_dim + self.distance_dim

            # gi, gj, gi*gj, distance between gi and gj
            self.gij_dim = self.gi_dim * 3 + self.distance_dim

            logger.info(f"For Bi-LSTM encoder: span_dim is {self.gi_dim}, pairs_dim is {self.gij_dim}")
            self.encoder = DocumentEncoder(hidden_dim, char_filters)
            # This module is responsible for scoring individual spans (potential mentions) within the document.
            self.score_spans = MentionScore(self.gi_dim, attn_dim, self.distance_dim)
            # This module is responsible for scoring pairs of spans to determine if they refer to the same entity.
            self.score_pairs = PairwiseScore(self.gij_dim, distance_dim)
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

    def forward(self, doc):
        """         Encode document
                    Predict unary mention scores, prune them
                    Predict pairwise coreference scores
        """

        # Encode the document, keep the LSTM hidden states and embedded tokens
        # states == These are the hidden states from the LSTM,
        # which capture the sequential and contextual information of the document.
        # embeds == These are the original token embeddings,
        # which are dense vector representations of the tokens without contextual information.
        logger.info(f"Encode document {doc}")
        states, embeds = self.encoder(doc)

        pass
        # # Get mention scores for each span, prune
        # spans, g_i, mention_scores = self.score_spans(states, embeds, doc)
        #
        # # Get pairwise scores for each span combo
        # spans, coref_scores = self.score_pairs(spans, g_i, mention_scores)
        #
        # return spans, coref_scores




class Trainer:
    """ Class dedicated to training and evaluating the model
    """

    def __init__(self, model, train_corpus, test_corpus,
                 steps, lr=1e-3):
        self.model = to_cuda(model)

        # create train corpus and eval corpus
        random.seed(42)
        split_idx = int(0.9 * len(train_corpus.docs))
        self.train_corpus = train_corpus[:split_idx]
        self.val_corpus = train_corpus[split_idx:]

        self.test_corpus = test_corpus
        self.steps = steps
        self.lr = lr

        self.optimizer = optim.Adam(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=100,
                                                   gamma=0.001)  # adjusts the learning rate during training

    def train(self, num_epochs, eval_interval=10, *args, **kwargs):
        """ Training  the model """

        for epoch in range(1, num_epochs + 1):
            self.train_epoch(epoch, *args, **kwargs)  # training each epoch


            logger.info(" Start saving the model.")
            self.save_model(str(datetime.now()))  # save the model with a filename based on the current date and time.

            # Evaluate every eval_interval epochs
            if epoch % eval_interval == 0:
                logger.info(" Evaluating every 10 epoch...")
                print('\n\nEVALUATION\n\n')
                self.model.eval()  # Sets the model to evaluation mode.

                #  Evaluates the model on the validation corpus and stores the results.
                results = self.evaluate(self.val_corpus)
                print(results)

    def train_epoch(self, epoch):
        """ Run a training epoch over 'steps' documents """
        # Set model to train (enables dropout)
        logger.info("Training epoch based on batches beginning...")
        self.model.train()

        # Randomly sample documents from the train corpus
        batch = random.sample(self.train_corpus, self.steps)

        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []

        for document in tqdm(batch):
            # Randomly truncate document to up to 50 sentences

            doc = document.truncate()  # discard docs with more than 50 sentences

            # Compute loss, number gold links found, total gold links
            loss, mentions_found, total_mentions, \
                corefs_found, total_corefs, corefs_chosen = self.train_doc(doc)
            # to compute the loss and various metrics for the truncated document.

            # Track stats by document for debugging
            print(document, '| Loss: %f | Mentions: %d/%d | Coref recall: %d/%d | Corefs precision: %d/%d' \
                  % (loss, mentions_found, total_mentions,
                     corefs_found, total_corefs, corefs_chosen, total_corefs))

            epoch_loss.append(loss)  # Adds the document's loss to the epoch_loss list.
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            epoch_corefs.append(safe_divide(corefs_found, total_corefs))
            epoch_identified.append(safe_divide(corefs_chosen, total_corefs))

        # Step the learning rate decrease scheduler
        self.scheduler.step()

        print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref precision: %f' \
              % (epoch, np.mean(epoch_loss), np.mean(epoch_mentions),
                 np.mean(epoch_corefs), np.mean(epoch_identified)))

    def train_doc(self, document):
        """
        Compute loss for a forward pass over a document
        """
        gold_corefs, total_corefs, \
            gold_mentions, total_mentions = extract_gold_corefs(document)

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        mentions_found, corefs_found, corefs_chosen = 0, 0, 0

        # Predict coref probabilities for each span in a document
        spans, probs = self.model(document)
        #  spans == These are the spans (segments) of text that the model identifies as potential coreference mentions.
        # probs == These are the probabilities associated with each span,
        # indicating the model's confidence that the span is a coreference mention.

        # Get log-likelihood of correct antecedents implied by gold clustering
        # gold_indexes = to_cuda(torch.zeros_like(probs))
        # for idx, span in enumerate(spans):
        #
        #     # Log number of mentions found
        #     if (span.i1, span.i2) in gold_mentions:
        #         mentions_found += 1
        #
        #         # Check which of these tuples are in the gold set, if any
        #         golds = [
        #             i for i, link in enumerate(span.yi_idx)
        #             if link in gold_corefs
        #         ]
        #
        #         # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
        #         if golds:
        #             gold_indexes[idx, golds] = 1
        #
        #             # Progress logging for recall
        #             corefs_found += len(golds)
        #             found_corefs = sum((probs[idx, golds] > probs[idx, len(span.yi_idx)])).detach()
        #             corefs_chosen += found_corefs.item()
        #         else:
        #             # Otherwise, set gold to dummy
        #             gold_indexes[idx, len(span.yi_idx)] = 1
        #
        # # Negative marginal log-likelihood
        # eps = 1e-8
        # loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1 - eps), dim=0) * -1)
        #
        # # Backpropagate
        # loss.backward()
        #
        # # Step the optimizer
        # self.optimizer.step()
        #
        # return (loss.item(), mentions_found, total_mentions,
        #         corefs_found, total_corefs, corefs_chosen)

        pass


if __name__ == "__main__":
    # Create coreference resolution model

    # embeds_dim = the dimensionality of token embeddings
    # hidden dim = the hidden dim of LSTM

    # Encoder type === can choose between ['lstm','HooshvareLab/bert-fa-zwnj-base']
    logger.info("Creating coref model...")
    model = CorefModel(embed_dim=400, hidden_dim=200, encoder_type='lstm')

    logger.info("Reading training and test corpora...")
    train_corpus = read_corpus('data/Mehr/train-dev/')
    test_corpus = read_corpus('data/Mehr/test/')

    # ?? train for 150 epochs, each  train 100 documents each doc up to 50 sentences for lstm
    trainer = Trainer(model, train_corpus, test_corpus, steps=5)

    logger.info("Training and test corpora loaded successfully.")
    trainer.train(150)

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
from subprocess import Popen, PIPE
import networkx as nx

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
        self.emb_dropout = nn.Dropout(0.50)
        self.lstm_dropout = nn.Dropout(0.20)  # Applied to the outputs of the LSTM layers.

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
        packed_data = self.emb_dropout(packed.data)
        packed = torch.nn.utils.rnn.PackedSequence(packed_data, packed.batch_sizes, packed.sorted_indices,
                                                   packed.unsorted_indices)

        # Pass an LSTM over the embeds
        output, _ = self.lstm(packed)

        # Apply dropout
        output_data = self.lstm_dropout(output.data)
        output = torch.nn.utils.rnn.PackedSequence(output_data, output.batch_sizes, output.sorted_indices,
                                                   output.unsorted_indices)

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
        """
        Compute unary mention score for each span
        """
        # Initialize Span objects containing start index, end index
        spans = [Span(i1=i[0], i2=i[-1], id=idx) for idx, i in enumerate(compute_idx_spans(doc.sents))]

        # Compute first part of attention over span states (alpha_t)
        attns = self.attention(states)

        # Regroup attn values, embeds into span representations
        # TODO: figure out a way to batch
        span_attns, span_embeds = zip(*[(attns[s.i1:s.i2 + 1], embeds[s.i1:s.i2 + 1])
                                        for s in spans])

        # Pad and stack span attention values, span embeddings for batching
        padded_attns, _ = pad_and_stack(span_attns, value=-1e10)
        padded_embeds, _ = pad_and_stack(span_embeds)

        # Weight attention values using softmax
        attn_weights = F.softmax(padded_attns, dim=1)

        # Compute self-attention over embeddings (x_hat)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        # Compute span widths (i.e. lengths), embed them
        widths = self.width([len(s) for s in spans])

        # Get LSTM state for start, end indexes
        start_end = torch.stack([torch.cat((states[s.i1], states[s.i2]))
                                 for s in spans])

        # Cat it all together to get g_i, our span representation
        g_i = torch.cat((start_end, attn_embeds, widths), dim=1)

        # Compute each span's unary mention score
        mention_scores = self.score(g_i)

        # Update span object attributes
        # (use detach so we don't get crazy gradients by splitting the tensors)
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores.detach())
        ]

        # Prune down to LAMBDA*len(doc) spans
        spans = prune(spans, len(doc))

        # Update antencedent set (yi) for each mention up to K previous antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx - K):idx])
            for idx, span in enumerate(spans)
        ]

        return spans, g_i, mention_scores


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module
    """

    def __init__(self, gij_dim, distance_dim):
        super().__init__()

        self.distance = Distance(distance_dim)
        self.score = Score(gij_dim)

    def forward(self, spans, g_i, mention_scores):
        """ Compute pairwise score for spans and their up to K antecedents """

        # Extract raw features
        mention_ids, antecedent_ids, distances = zip(*[
            (i.id, j.id, i.i2 - j.i1)
            for i in spans
            for j in i.yi
        ])

        # For indexing a tensor efficiently
        mention_ids = to_cuda(torch.tensor(mention_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))

        # Embed them
        phi = self.distance(distances)

        # Extract their span representations from the g_i matrix
        i_g = torch.index_select(g_i, 0, mention_ids)
        j_g = torch.index_select(g_i, 0, antecedent_ids)

        # Create s_ij representations
        pairs = torch.cat((i_g, j_g, i_g * j_g, phi), dim=1)

        # Extract mention score for each mention and its antecedents
        s_i = torch.index_select(mention_scores, 0, mention_ids)
        s_j = torch.index_select(mention_scores, 0, antecedent_ids)

        # Score pairs of spans for coreference link
        s_ij = self.score(pairs)

        # Compute pairwise scores for coreference links between each mention and its antecedents
        coref_scores = torch.sum(torch.cat((s_i, s_j, s_ij), dim=1), dim=1, keepdim=True)

        # Update spans with set of possible antecedents' indices, scores
        spans = [
            attr.evolve(span,
                        yi_idx=[((y.i1, y.i2), (span.i1, span.i2)) for y in span.yi]
                        )
            for span, score, (i1, i2) in zip(spans, coref_scores, pairwise_indexes(spans))
        ]

        # Get antecedent indexes for each span
        antecedent_idx = [len(s.yi) for s in spans if len(s.yi)]

        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores = [to_cuda(torch.tensor([]))] + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(torch.tensor([[0.]]))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        # Batch and softmax
        probs = [F.softmax(tensr) for tensr in with_epsilon]

        # Pad the scores for each one with a dummy value, 1000 so that the tensors can
        # be of the same dimension for calculation loss and what not.
        probs, _ = pad_and_stack(probs, value=1000)
        probs = probs.squeeze()

        return spans, probs


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
        logger.info(f"Encode document {doc.filename}")
        states, embeds = self.encoder(doc)

        # Get mention scores for each span, prune
        # spans: The spans of text (sub-sequences of tokens) identified as potential mentions.
        # g_i: The span representations (features) for each identified span.
        # mention_scores: The scores for each span, indicating the likelihood that the span is a mention.
        # logger.info(f"Calculate mention scores for document {doc.filename}")
        spans, g_i, mention_scores = self.score_spans(states, embeds, doc)

        # Get pairwise scores for each span combo
        # logger.info(f"Calculate pairwise scores for document {doc.filename}")
        spans, coref_scores = self.score_pairs(spans, g_i, mention_scores)

        return spans, coref_scores


class Trainer:
    """ Class dedicated to training and evaluating the model
    """

    def __init__(self, model, train_corpus, test_corpus, dev_corpus,
                 steps, lr=1e-3):
        self.model = to_cuda(model)

        self.train_corpus = list(train_corpus)

        self.val_corpus = dev_corpus

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

    def train(self, num_epochs, eval_interval=1, *args, **kwargs):
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
        #  spans == These are the spans (segments) of text that the model identifies as potential coreference mentions.
        # probs == These are the probabilities associated with each span,
        # indicating the model's confidence that the span is a coreference mention.
        spans, probs = self.model(document)

        pass

        # Get log-likelihood of correct antecedents implied by gold clustering
        gold_indexes = to_cuda(torch.zeros_like(probs))
        for idx, span in enumerate(spans):

            # Log number of mentions found
            if (span.i1, span.i2) in gold_mentions:
                mentions_found += 1

                # Check which of these tuples are in the gold set, if any
                golds = [
                    i for i, link in enumerate(span.yi_idx)
                    if link in gold_corefs
                ]

                # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
                if golds:
                    gold_indexes[idx, golds] = 1

                    # Progress logging for recall
                    corefs_found += len(golds)
                    found_corefs = sum((probs[idx, golds] > probs[idx, len(span.yi_idx)])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    # Otherwise, set gold to dummy
                    gold_indexes[idx, len(span.yi_idx)] = 1

        # Negative marginal log-likelihood
        eps = 1e-8
        loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp(min=eps, max=1 - eps)) * -1)

        pass

        # Backpropagate
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, corefs_chosen)

        pass

    def save_model(self, savepath):
        """ Save model state dictionary """
        model_dir = "data/model/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        valid_savepath = os.path.join(model_dir, savepath.replace(":", "-") + '.pth')
        torch.save(self.model.state_dict(), valid_savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)

    def evaluate(self, val_corpus, eval_script='eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        # Predict files
        print('Evaluating on validation corpus...')
        predicted_docs = [self.predict(doc) for doc in tqdm(val_corpus)]
        val_corpus.docs = predicted_docs

        # Output results
        golds_file, preds_file = self.to_conll(val_corpus, eval_script)

        # Run perl script
        print('Running Perl evaluation script...')
        p = Popen(['perl', eval_script, 'all', golds_file, preds_file], stdout=PIPE)
        stdout, stderr = p.communicate()
        results = str(stdout).split('TOTALS')[-1]

        # Write the results out for later viewing
        with open('data/preds/results.txt', 'w+') as f:
            f.write(results)
            f.write('\n\n\n')

        return results

    def predict(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()

        # Pass the document through the model
        spans, probs = self.model(doc)

        # Cluster found coreference links
        for i, span in enumerate(spans):

            # Loss implicitly pushes coref links above 0, rest below 0
            found_corefs = [idx
                            for idx, _ in enumerate(span.yi_idx)
                            if probs[i, idx] > probs[i, len(span.yi_idx)]]

            # If we have any
            if any(found_corefs):

                # Add edges between all spans in the cluster
                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    graph.add_edge((span.i1, span.i2), (link.i1, link.i2))

        # Extract clusters as nodes that share an edge
        clusters = list(nx.connected_components(graph))

        # Initialize token tags
        token_tags = [[] for _ in range(len(doc))]

        # Add in cluster ids for each cluster of corefs in place of token tag
        for idx, cluster in enumerate(clusters):
            for i1, i2 in cluster:

                if i1 == i2:
                    token_tags[i1].append(f'({idx})')

                else:
                    token_tags[i1].append(f'({idx}')
                    token_tags[i2].append(f'{idx})')

        doc.tags = ['|'.join(t) if t else '-' for t in token_tags]

        return doc

    def to_conll(self, val_corpus, eval_script):
        """ Write to out_file the predictions, return CoNLL metrics results """

        # Make predictions directory if there isn't one already
        golds_file, preds_file = 'data/preds/golds.txt', 'data/preds/predictions.txt'
        if not os.path.exists('data/preds/'):
            os.makedirs('data/preds/')

        # Combine all gold files into a single file (Perl script requires this)
        golds_file_content = flatten([doc.raw_text for doc in val_corpus])
        with io.open(golds_file, 'w', encoding='utf-8', errors='strict') as f:
            for line in golds_file_content:
                f.write(line)

        # Dump predictions
        with io.open(preds_file, 'w', encoding='utf-8', errors='strict') as f:

            for doc in val_corpus:

                current_idx = 0

                for line in doc.raw_text:

                    # Indicates start / end of document or line break
                    if line.startswith('#begin') or line.startswith('#end') or line == '\n':
                        f.write(line)
                        continue
                    else:
                        # Replace the coref column entry with the predicted tag
                        tokens = line.split()
                        tokens[-1] = doc.tags[current_idx]

                        # Increment by 1 so tags are still aligned
                        current_idx += 1

                        # Rewrite it back out
                        f.write('\t'.join(tokens))
                    f.write('\n')

        return golds_file, preds_file


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

    # Split the train_corpus into train and dev sets
    train_corpus, dev_corpus = train_corpus.split_corpus()

    # ?? train for 150 epochs, each  train 100 documents each doc up to 50 sentences for lstm
    trainer = Trainer(model, train_corpus, test_corpus, dev_corpus, steps=1)

    logger.info("Training and test corpora loaded successfully.")
    trainer.train(150)

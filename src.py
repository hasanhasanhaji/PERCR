import logging
import configparser
import random
import os
import io
from tqdm import tqdm
import torch.optim as optim
from conll import read_corpus, LazyVectors
from utils import *
from datetime import datetime
from subprocess import Popen, PIPE
import networkx as nx
from model import CorefModel

# configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# initialize configuration file
config = configparser.ConfigParser()
config.read('config.ini')  # Load configuration from file


class Trainer:
    """ Class dedicated to training and evaluating the model
    """

    def __init__(self, final_model, train, test, val,
                 step, lr=config.getfloat('TRAINING', 'lr')):

        self.model = to_cuda(final_model)
        self.train_corpus = list(train)
        self.val_corpus = val
        self.test_corpus = test
        self.steps = step
        self.lr = lr
        self.optimizer = optim.Adam(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )
        # adjusts the learning rate during training
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=config.getint('TRAINING', 'scheduler_step_size'),
                                                   gamma=config.getfloat('TRAINING', 'scheduler_gamma'))

    def train(self, num_epochs, eval_interval=config.getint('TRAINING', 'eval_interval'), *args, **kwargs):
        """
        Training  the coref model
        """

        for epoch in range(1, num_epochs + 1):
            logger.info("Training epoch based on batches beginning...")
            self.model.train()  # Set model to train (enables dropout)

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

                # Tracking and Logging Per-Document Statistics
                # Loss: The loss value for the document, indicating how well the model's predictions
                # match the true coreference.
                # Mentions: %d/%d: The number of mentions the model found correctly (mentions_found / total_mentions)
                # Coref recall: %d/%d The number of coreference links the model found correctly (corefs_found)
                # out of the total number of true coreference links (total_corefs).
                # Corefs precision: %d/%d: The number of coreference links the model predicted correctly
                # (corefs_chosen) out of the total number of coreference links the model predicted (total_corefs).
                print(document, '| Loss: %f | Mentions: %d/%d | Coref recall: %d/%d | Corefs precision: %d/%d' \
                      % (loss, mentions_found, total_mentions,
                         corefs_found, total_corefs, corefs_chosen, total_corefs))

                epoch_loss.append(loss)  # Adds the document's loss to the epoch_loss list.
                # Stores the mention recall for all documents in the epoch.
                epoch_mentions.append(safe_divide(mentions_found, total_mentions))
                #  Stores the coreference recall for all documents in the epoch.
                epoch_corefs.append(safe_divide(corefs_found, total_corefs))
                # Stores the coreference precision for all documents in the epoch.
                epoch_identified.append(safe_divide(corefs_chosen, total_corefs))

            # Step the learning rate decrease scheduler
            self.scheduler.step()

            # Loss = The average loss over all documents processed in the epoch.
            # Mention recall= The average recall for mention detection
            # Coref recall: The average recall for coreference linking
            # Coref precision: The average precision for coreference linking
            print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref precision: %f' \
                  % (epoch, np.mean(epoch_loss), np.mean(epoch_mentions),
                     np.mean(epoch_corefs), np.mean(epoch_identified)))

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

        # Backpropagation
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, corefs_chosen)

    def save_model(self, savepath):
        """ Save model state dictionary """
        model_dir = config.get('DATA', 'model_address')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        valid_savepath = os.path.join(model_dir, savepath.replace(":", "-") + '.pth')
        torch.save(self.model.state_dict(), valid_savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        try:
            state_dict = torch.load(loadpath)
            self.model.load_state_dict(state_dict)
            self.model = to_cuda(self.model)
            logger.info(f"Model successfully loaded from {loadpath}")
        except FileNotFoundError:
            logger.warning(f"No model found at {loadpath}. Starting training from scratch.")
        except Exception as e:  # Catching broader exceptions for robustness
            logger.error(f"Error loading model: {e}. Starting training from scratch.")

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
        p = Popen([eval_script, 'all', golds_file, preds_file], stdout=PIPE)
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


def get_latest_model_path(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if models:
        latest_model = max(models, key=lambda f: os.path.getctime(os.path.join(model_dir, f)))
        return os.path.join(model_dir, latest_model)
    else:
        return None


if __name__ == "__main__":
    corpus_type = config.get('DATA', 'corpus_type')

    logger.info("Reading training and test corpora...")
    # Read corpus paths from configuration based on selected corpus type
    train_corpus_path = config.get('DATA', f'{corpus_type}_corpus_path_train')
    test_corpus_path = config.get('DATA', f'{corpus_type}_corpus_path_test')
    train_corpus = read_corpus(train_corpus_path, corpus_type)
    test_corpus = read_corpus(test_corpus_path, corpus_type)

    # Share the vocabulary for both GLOVE and W2VEC
    corpus_vocab = train_corpus.vocab
    corpus_char_vocab = train_corpus.char_vocab

    GLOVE = LazyVectors.from_corpus(corpus_vocab, name='glove_arman_300.txt', cache='data/vectors/')
    W2VEC = LazyVectors.from_corpus(corpus_vocab, name='word2vec_wikipedia_50.txt', cache='data/vectors/')

    # Split the train_corpus into train and dev sets
    train_corpus, dev_corpus = train_corpus.split_corpus()
    logger.info("Training and test corpora loaded successfully.")

    # Create coreference resolution model
    logger.info("Creating coref model...")
    model = CorefModel(
        embed_dim=config.getint('MODEL', 'embed_dim'),
        hidden_dim=config.getint('MODEL', 'hidden_dim'),
        encoder_type=config.get('MODEL', 'encoder_type'),
        char_vocab=corpus_char_vocab,
        GLOVE=GLOVE,
        W2VEC=W2VEC)

    # Determine the steps value based on the dataset in use
    steps = config.getint('TRAINING', 'mehr_steps') if corpus_type == "Mehr" \
        else config.getint('TRAINING', 'rcdat_steps')

    # train for 150 epochs, each  train ? documents(steps) each doc up to 50 sentences for lstm
    trainer = Trainer(model, train_corpus, test_corpus, dev_corpus, steps)

    # Check for existing model to resume training
    model_dir = config.get('DATA', 'model_address')
    existing_model = get_latest_model_path(model_dir)  # Function to find latest .pth file
    if existing_model:
        trainer.load_model(existing_model)  # Load the model if found

    trainer.train(config.getint('TRAINING', 'epochs'))

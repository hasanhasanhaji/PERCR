import logging
import torch
import torch.nn as nn

# configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


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
            # self.encoder = DocumentEncoder(embed_dim, char_filters)
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

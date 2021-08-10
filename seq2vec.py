"""Word2Vec embedding neural network for sequence-based recommendations."""

import logging
from typing import List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


def generate_sequences(
    interaction_data: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    timestamp_col: str = "timestamp",
    return_as_array: bool = False,
) -> List:
    """Generates sequences from interaction data.

    To generate the sequences an array is prefilled with -1 which serves as padding.
    Once the sequence array has been populated with the actual sequences it is
    transformed to a list and the -1 padding is filtered.

    (Note: np.lexsort uses the final argument as primary key. It sorts first by user
    and then by timestamp.)

    Args:
        interaction_data: Interaction data with encoded user-item pairs.
        user_col: Encoded user identifier.
        item_col: Encoded item identifier.
        timestamp_col: Time of interaction.

    Returns:
        List: List of sequences.
    """
    logger.info("Generating sequences from encoded interaction data.")
    interaction_data[item_col] = interaction_data[item_col]
    item_ids = interaction_data[item_col].values
    user_ids = interaction_data[user_col].values
    timestamps = interaction_data[timestamp_col].values

    sorting_index = np.lexsort((timestamps, user_ids))
    user_ids = user_ids[sorting_index]
    item_ids = item_ids[sorting_index]

    unique_user_ids, index, counts = np.unique(
        user_ids, return_index=True, return_counts=True
    )

    n_sequences = len(unique_user_ids)
    max_sequence_len = counts.max()
    sequences = np.zeros((n_sequences, max_sequence_len), dtype=np.int32) - 1
    logger.info(
        f"Generating {n_sequences} sequences with max length {max_sequence_len}"
    )

    for i, (sequence_start, sequence_end, sequence_len) in tqdm(
        enumerate(zip(index, index + counts, counts))
    ):
        sequences[i, :sequence_len] = item_ids[sequence_start:sequence_end]

    if return_as_array:
        return sequences

    sequences = sequences.tolist()  # strip -1 padding from all sequences
    sequences = [[item for item in seq if item != -1] for seq in sequences]

    return sequences


def embedding_model(
    n_factors: int = 50,
    window: int = 5,
    min_count: int = 1,
    learning_rate: float = 0.05,
    negative_samples: int = 10,
    negative_exponent: float = 0.75,
    workers: int = 4,
    n_iterations: int = 10,
    batch_size: int = 10000,
    skip_gram: int = 0,
) -> Word2Vec:
    """Neural network for training item embedding vectors based on Word2Vec.

    Word2Vec is a model that embeds items in a lower-dimensional vector space using a
    neural network. The result is a set of item vectors where vectors close together in
    vector space are similar based on context (the sequence in which they appear).

    Supports both Skip-Gram and CBOW (Continuous Bag of Words) algorithms for training.
    Skip-Gram works well on smaller data, and can better represent less frequent items.
    CBOW trains faster than Skip-Gram, and can better represent more frequent items.
    In a recommender system context; CBOW will lean more towards popular items and
    Skip-Gram will lean more towards exploration of new items.

    For a reference on choosing training algorithm and setting hyperparameters for
    recommendation purposes see this paper from Twitter:
    Chamberlain, B, et al. "Tuning Word2vec for Large Scale Recommendation Systems."
    https://arxiv.org/pdf/2009.12192.pdf

    More details on the algorithm can be found here:
    https://code.google.com/p/word2vec/.

    Args:
        n_factors: Number of latent factors for the low-rank embedding vectors.
        window: Maximum distance between the current and predicted item in a sequence.
        min_count: Ignores all items with total frequency lower than this.
        learning_rate: The initial learning rate for gradient updates.
        negative_samples: If > 0, negative sampling will be used, the int for negative
            specifies how many negative items should be drawn (usually between 5-20).
        workers: Use these many worker threads in parallel to train the model.
        n_iterations: Number of training iterations over all sequences.
        batch_size: Target size (in items) for batches of examples passed to workers.
        skip_gram: Training algorithm to use: 1 for skip-gram; otherwise CBOW.

    Returns:
        Word2Vec: Embedding neural network model.
    """
    logger.info("Defining Embedding Neural Network model.")
    model = Word2Vec(
        vector_size=n_factors,
        window=window,
        min_count=min_count,
        alpha=learning_rate,
        negative=negative_samples,
        ns_exponent=negative_exponent,
        workers=workers,
        epochs=n_iterations,
        batch_words=batch_size,
        sg=skip_gram,
        compute_loss=True,
    )
    return model


def fit(model: Word2Vec, sequences: List):
    """Trains the defined embedding neural network model.

    Args:
        model: Embedding neural network model
        sequences: List of sequences.

    Returns:
        Word2Vec: Trained Embedding neural network model.
    """
    logger.info("Building item vocabulary for training.")
    model.build_vocab(sequences, progress_per=1000, update=False)
    logger.info("Fitting Embedding Neural Network model.")
    model.train(sequences, epochs=model.epochs, total_examples=model.corpus_count)
    training_loss = model.get_latest_training_loss()
    logger.info(f"Final model training loss: {training_loss}")
    return model


def get_item_embeddings(model: Word2Vec) -> np.ndarray:
    """Extracts the item embedding vectors from the trained model.

    Args:
        model: Embedding neural network model

    Returns:
        np.ndarray: Item embedding vectors.
    """
    logger.info("Getting item embeddings.")
    item_embeddings = model.wv.get_normed_vectors()
    item_embeddings = np.array(item_embeddings)
    return item_embeddings


def get_user_embeddings(sequences: List, item_embeddings: np.ndarray) -> np.ndarray:
    """Generates user embeddings by summarizing their sequences of item embeddings.

    A user embedding is created by averaging all item embedding vectors in the sequence.
    This leads to a single user embedding based on his or her implicit feedback.

    Args:
        sequences: List of sequences.
        np.ndarray: Item embedding vectors.

    Returns:
        np.ndarray: User embedding vectors.
    """
    logger.info("Getting user embeddings.")
    user_embeddings = [
        np.mean(item_embeddings[sequences[i]], axis=0) for i in range(len(sequences))
    ]
    user_embeddings = np.array(user_embeddings)
    return user_embeddings


def compute_precision(recommendations: np.ndarray, sequences: np.ndarray):
    """Compute average precision of recommendations.

    Args:
        recommendations: Array of top N item recommendations for each user.
        sequences: Array of item sequences for each user.

    Returns:
        Average precision.
    """
    precision = [
        len(
            np.intersect1d(
                recommendations[seq], np.unique(sequences[seq]), assume_unique=True,
            )
        )
        / np.min([len(np.unique(sequences[seq])), 10])
        for seq in trange(len(recommendations))
    ]
    return np.mean(precision)

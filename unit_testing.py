import pandas as pd
import pytest
from gensim.models import Word2Vec

from .seq2vec import (
    compute_precision,
    embedding_model,
    fit,
    generate_sequences,
    get_item_embeddings,
    get_user_embeddings,
)


@pytest.fixture
def interaction_data() -> pd.DataFrame:
    interaction_data = pd.DataFrame(
        {
            "user": [111, 222, 111, 333, 222, 333, 444, 444],
            "item": ["A", "C", "A", "B", "C", "D", "A", "D"],
            "views": [1, 3, 2, 1, 1, 2, 1, 5],
            "user_id": [0, 1, 0, 2, 1, 2, 3, 3],
            "item_id": [0, 2, 1, 1, 4, 3, 0, 3],
            "interaction": [1, 1, 1, 1, 1, 1, 1, 1],
            "timestamp": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    return interaction_data


def test_generate_sequences(interaction_data):
    sequences = generate_sequences(
        interaction_data,
        user_col="user_id",
        item_col="item_id",
        timestamp_col="timestamp",
    )
    assert sequences == [[0, 1], [2, 4], [1, 3], [0, 3]]


def test_embedding_model():
    model = embedding_model(
        n_factors=6,
        window=2,
        negative_samples=1,
        n_iterations=1,
        batch_size=1,
        workers=1,
    )
    assert isinstance(model, Word2Vec)


def test_fit():
    sequences = [[0, 1], [2, 4], [1, 3], [0, 3]]
    model = embedding_model(n_factors=6, n_iterations=1)
    trained_model = fit(model, sequences)
    assert isinstance(trained_model, Word2Vec)


def test_get_item_embeddings():
    sequences = [[0, 1], [2, 4], [1, 3], [0, 3]]
    model = embedding_model(n_factors=6, n_iterations=1)
    trained_model = fit(model, sequences)
    item_embeddings = get_item_embeddings(trained_model)
    assert item_embeddings.shape == (5, 6)


def test_get_user_embeddings():
    sequences = [[0, 1], [2, 4], [1, 3], [0, 3]]
    model = embedding_model(n_factors=6, n_iterations=1)
    trained_model = fit(model, sequences)
    item_embeddings = get_item_embeddings(trained_model)
    user_embeddings = get_user_embeddings(sequences, item_embeddings)
    assert user_embeddings.shape == (4, 6)


def test_compute_precision():
    recommendations = [[1], [2], [1], [3]]
    sequences = [[0, 1], [2, 4], [1, 3], [0, 3]]
    precision = compute_precision(recommendations, sequences)
    assert precision == 0.5

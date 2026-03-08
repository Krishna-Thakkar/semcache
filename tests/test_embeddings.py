import numpy as np
import pytest
from semcache.embeddings.sentence_transformer import SentenceTransformerEmbedding


@pytest.fixture(scope="module")
def embedding():
    return SentenceTransformerEmbedding()


class TestSentenceTransformerEmbedding:
    def test_model_loads(self):
        engine = SentenceTransformerEmbedding()
        assert engine is not None

    def test_single_embed_shape(self, embedding):
        vector = embedding.embed("explain cnn")
        assert vector.shape == (384,)

    def test_single_embed_dtype(self, embedding):
        vector = embedding.embed("explain cnn")
        assert vector.dtype == np.float32

    def test_single_embed_normalized(self, embedding):
        vector = embedding.embed("explain cnn")
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-5

    def test_batch_embed_shape(self, embedding):
        vectors = embedding.embed_batch(["explain cnn", "what is cnn"])
        assert vectors.shape == (2, 384)

    def test_batch_embed_dtype(self, embedding):
        vectors = embedding.embed_batch(["explain cnn", "what is cnn"])
        assert vectors.dtype == np.float32

    def test_batch_embed_normalized(self, embedding):
        vectors = embedding.embed_batch(["explain cnn", "what is cnn"])
        for i in range(vectors.shape[0]):
            norm = np.linalg.norm(vectors[i])
            assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_high_similarity(self, embedding):
        v1 = embedding.embed("explain cnn")
        v2 = embedding.embed("what is a convolutional neural network")
        similarity = float(np.dot(v1, v2))
        assert similarity > 0.5

    def test_dissimilar_texts_lower_similarity(self, embedding):
        v1 = embedding.embed("explain cnn")
        v2 = embedding.embed("what is the capital of france")
        similarity_similar = float(np.dot(
            embedding.embed("explain cnn"),
            embedding.embed("what is a convolutional neural network")
        ))
        similarity_dissimilar = float(np.dot(v1, v2))
        assert similarity_similar > similarity_dissimilar

from app.embeddings import HashingEmbeddingProvider


def test_embedding_is_deterministic():
    provider = HashingEmbeddingProvider(dim=64)
    [a] = provider.embed(["hello world"])
    [b] = provider.embed(["hello world"])
    assert a == b


def test_embedding_is_unit_normalized():
    provider = HashingEmbeddingProvider(dim=64)
    [vec] = provider.embed(["some reasonably long sentence to embed"])
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_different_text_gives_different_vectors():
    provider = HashingEmbeddingProvider(dim=64)
    vecs = provider.embed(["apples and oranges", "quantum computing basics"])
    assert vecs[0] != vecs[1]

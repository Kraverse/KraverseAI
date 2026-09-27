from app.rag import chunk_text, cosine_similarity


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_splits():
    chunks = chunk_text("a" * 2500, size=1000, overlap=100)
    assert len(chunks) >= 3
    assert all(chunks)


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0

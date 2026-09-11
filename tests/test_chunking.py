from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_single_chunk():
    chunks = chunk_text("one two three", chunk_size_tokens=10, overlap_tokens=2)
    assert len(chunks) == 1
    assert chunks[0].text == "one two three"


def test_overlap_shares_tokens_between_chunks():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size_tokens=10, overlap_tokens=3)
    assert len(chunks) >= 2
    first_tail = chunks[0].text.split()[-3:]
    second_head = chunks[1].text.split()[:3]
    assert first_tail == second_head


def test_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size_tokens=5, overlap_tokens=5)

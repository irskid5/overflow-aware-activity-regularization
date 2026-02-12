"""Tests for PTB data loading."""
import numpy as np
import pytest


def test_download_ptb_returns_directory_path():
    """download_ptb returns path to directory containing PTB files."""
    from experiments.ptb.data import download_ptb

    data_dir = download_ptb()
    assert data_dir is not None
    assert isinstance(data_dir, str)


def test_ptb_files_exist_after_download():
    """All three PTB files exist after download."""
    import os
    from experiments.ptb.data import download_ptb

    data_dir = download_ptb()
    assert os.path.exists(os.path.join(data_dir, "ptb.train.txt"))
    assert os.path.exists(os.path.join(data_dir, "ptb.valid.txt"))
    assert os.path.exists(os.path.join(data_dir, "ptb.test.txt"))


def test_read_words_returns_list():
    """read_words returns list of strings."""
    import os
    from experiments.ptb.data import download_ptb, read_words

    data_dir = download_ptb()
    words = read_words(os.path.join(data_dir, "ptb.train.txt"))
    assert isinstance(words, list)
    assert len(words) > 0
    assert all(isinstance(w, str) for w in words[:100])


def test_read_words_contains_eos():
    """read_words includes <eos> tokens for newlines."""
    import os
    from experiments.ptb.data import download_ptb, read_words

    data_dir = download_ptb()
    words = read_words(os.path.join(data_dir, "ptb.train.txt"))
    assert "<eos>" in words


def test_train_has_approximately_929k_words():
    """Training set has approximately 929k words."""
    import os
    from experiments.ptb.data import download_ptb, read_words

    data_dir = download_ptb()
    words = read_words(os.path.join(data_dir, "ptb.train.txt"))
    # Allow 5% tolerance
    assert 880_000 < len(words) < 980_000


def test_build_vocab_returns_dict():
    """build_vocab returns word-to-id mapping."""
    from experiments.ptb.data import build_vocab

    words = ["the", "cat", "sat", "the", "cat", "<eos>"]
    vocab = build_vocab(words)
    assert isinstance(vocab, dict)
    assert "the" in vocab
    assert "cat" in vocab


def test_vocab_ids_are_unique():
    """Each word gets a unique ID."""
    from experiments.ptb.data import build_vocab

    words = ["the", "cat", "sat", "the", "cat", "<eos>"]
    vocab = build_vocab(words)
    ids = list(vocab.values())
    assert len(ids) == len(set(ids))


def test_vocab_sorted_by_frequency():
    """Most frequent words get lower IDs."""
    from experiments.ptb.data import build_vocab

    words = ["rare", "common", "common", "common", "rare"]
    vocab = build_vocab(words)
    assert vocab["common"] < vocab["rare"]


def test_ptb_vocab_approximately_10k():
    """PTB vocabulary is approximately 10k words."""
    import os
    from experiments.ptb.data import download_ptb, read_words, build_vocab

    data_dir = download_ptb()
    words = read_words(os.path.join(data_dir, "ptb.train.txt"))
    vocab = build_vocab(words)
    assert 9_000 < len(vocab) < 11_000


def test_producer_yields_correct_shapes():
    """ptb_producer yields (x, y) with correct shapes."""
    from experiments.ptb.data import ptb_producer

    data = list(range(100))
    batch_size = 4
    num_steps = 5

    batches = list(ptb_producer(data, batch_size, num_steps))
    assert len(batches) > 0

    x, y = batches[0]
    assert x.shape == (batch_size, num_steps)
    assert y.shape == (batch_size, num_steps)


def test_producer_y_is_x_shifted():
    """y is x shifted by 1 position."""
    from experiments.ptb.data import ptb_producer

    data = list(range(100))
    batches = list(ptb_producer(data, batch_size=2, num_steps=5))
    x, y = batches[0]
    
    # For batch=0, stream=0: x=[0,1,2,3,4], y=[1,2,3,4,5]
    assert y[0, 0] == x[0, 0] + 1
    assert y[0, -1] == x[0, -1] + 1


def test_producer_batches_are_consecutive():
    """Consecutive batches continue from where previous ended."""
    from experiments.ptb.data import ptb_producer

    data = list(range(100))
    batches = list(ptb_producer(data, batch_size=2, num_steps=5))
    
    x1, _ = batches[0]
    x2, _ = batches[1]
    
    # Second batch should continue where first ended
    assert x2[0, 0] == x1[0, -1] + 1


def test_producer_handles_empty_data():
    """ptb_producer returns empty list for empty data."""
    from experiments.ptb.data import ptb_producer

    batches = list(ptb_producer([], batch_size=2, num_steps=5))
    assert batches == []


def test_get_datasets_returns_four_items():
    """get_datasets returns (train, val, test, vocab_size)."""
    from experiments.ptb.data import get_datasets

    result = get_datasets(batch_size=20, num_steps=10)
    assert len(result) == 4


def test_datasets_have_correct_batch_shapes():
    """Dataset batches have correct shapes."""
    import tensorflow as tf
    from experiments.ptb.data import get_datasets

    batch_size = 20
    num_steps = 10
    train_ds, val_ds, test_ds, vocab_size = get_datasets(batch_size, num_steps)

    # Check one batch
    for x, y in train_ds.take(1):
        assert x.shape == (batch_size, num_steps)
        assert y.shape == (batch_size, num_steps)
        assert x.dtype == tf.int32
        assert y.dtype == tf.int32


def test_vocab_size_approximately_10k():
    """Vocabulary size is approximately 10k."""
    from experiments.ptb.data import get_datasets

    _, _, _, vocab_size = get_datasets(batch_size=20, num_steps=10)
    assert 9_000 < vocab_size < 11_000


def test_word_ids_within_vocab_range():
    """All word IDs are within [0, vocab_size)."""
    from experiments.ptb.data import get_datasets

    train_ds, _, _, vocab_size = get_datasets(batch_size=20, num_steps=10)

    for x, y in train_ds.take(5):
        assert x.numpy().min() >= 0
        assert x.numpy().max() < vocab_size
        assert y.numpy().min() >= 0
        assert y.numpy().max() < vocab_size

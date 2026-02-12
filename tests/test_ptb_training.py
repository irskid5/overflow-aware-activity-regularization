"""Tests for PTB training utilities."""
import dataclasses
import os

import pytest
import tensorflow as tf


def test_make_data_loader_returns_callable():
    """make_data_loader returns DataLoader-compatible callable."""
    from experiments.ptb.training import make_data_loader
    from experiments.ptb.config import BATCH_SIZE

    data_loader = make_data_loader()
    
    # Should accept (batch_size, enlarge) per DataLoader protocol
    # PTB ignores enlarge (not applicable)
    train_ds, val_ds, test_ds = data_loader(BATCH_SIZE, enlarge=False)
    
    assert train_ds is not None
    assert val_ds is not None
    assert test_ds is not None


def test_evaluate_perplexity():
    """evaluate_perplexity computes perplexity on dataset."""
    from experiments.ptb.training import evaluate_perplexity
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT, BATCH_SIZE, NUM_STEPS

    vocab_size = 100
    model = _get_model(PTB_EXPERIMENT.steps[1], vocab_size=vocab_size)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )

    # Create tiny dataset
    x = tf.random.uniform((BATCH_SIZE * 2, NUM_STEPS), maxval=vocab_size, dtype=tf.int32)
    y = tf.random.uniform((BATCH_SIZE * 2, NUM_STEPS), maxval=vocab_size, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE)

    ppl = evaluate_perplexity(model, dataset)

    assert isinstance(ppl, float)
    assert ppl > 0


def test_get_vocab_size():
    """get_vocab_size returns vocabulary size."""
    from experiments.ptb.training import get_vocab_size

    vocab_size = get_vocab_size()
    
    # PTB vocab is approximately 10k
    assert 9_000 < vocab_size < 11_000

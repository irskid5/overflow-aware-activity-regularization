# tests/test_perplexity.py
"""Tests for Perplexity metric."""
import numpy as np
import pytest
import tensorflow as tf


def test_perplexity_perfect_prediction():
    """Perplexity is 1.0 for perfect predictions (0 loss)."""
    from oar.metrics import Perplexity

    metric = Perplexity()
    
    # Perfect prediction: all probability on correct class
    # Use near-perfect (0.999) to avoid log(0) issues
    y_true = tf.constant([[0, 1, 2]])  # (1, 3)
    y_pred = tf.constant([
        [[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005], [0.0005, 0.0005, 0.999]]
    ])  # (1, 3, 3)
    
    metric.update_state(y_true, y_pred)
    result = metric.result().numpy()
    
    # exp(~0) ≈ 1.0
    assert np.isclose(result, 1.0, atol=0.05)


def test_perplexity_uniform_prediction():
    """Perplexity equals vocab_size for uniform predictions."""
    from oar.metrics import Perplexity

    metric = Perplexity()
    vocab_size = 100
    
    # Uniform prediction over vocab_size classes
    y_true = tf.constant([[0, 1, 2]])  # (1, 3)
    uniform = 1.0 / vocab_size
    y_pred = tf.fill([1, 3, vocab_size], uniform)
    
    metric.update_state(y_true, y_pred)
    result = metric.result().numpy()
    
    # exp(log(vocab_size)) = vocab_size
    assert np.isclose(result, vocab_size, rtol=0.01)


def test_perplexity_accumulates_across_batches():
    """Perplexity correctly accumulates across multiple batches."""
    from oar.metrics import Perplexity

    metric = Perplexity()
    
    # Two batches with uniform predictions
    vocab_size = 10
    uniform = 1.0 / vocab_size
    
    y_true1 = tf.constant([[0, 1]])
    y_pred1 = tf.fill([1, 2, vocab_size], uniform)
    metric.update_state(y_true1, y_pred1)
    
    y_true2 = tf.constant([[2, 3, 4]])
    y_pred2 = tf.fill([1, 3, vocab_size], uniform)
    metric.update_state(y_true2, y_pred2)
    
    result = metric.result().numpy()
    
    # Should still be vocab_size (average of same distribution)
    assert np.isclose(result, vocab_size, rtol=0.01)


def test_perplexity_reset_state():
    """reset_state clears accumulated values."""
    from oar.metrics import Perplexity

    metric = Perplexity()
    
    y_true = tf.constant([[0]])
    y_pred = tf.constant([[[0.5, 0.5]]])
    metric.update_state(y_true, y_pred)
    
    metric.reset_state()
    
    # After reset, total_loss and count should be 0
    assert metric.total_loss.numpy() == 0.0
    assert metric.count.numpy() == 0.0

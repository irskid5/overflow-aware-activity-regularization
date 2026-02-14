"""Tests for Speech Commands data loading."""
import tensorflow as tf


def test_get_datasets_returns_three_datasets():
    """get_datasets returns train, val, test datasets."""
    from experiments.speech_commands.data import get_datasets

    train_ds, val_ds, test_ds = get_datasets(batch_size=32)

    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)


def test_dataset_shapes():
    """Datasets have correct MFCC shapes: (batch, frames, mfcc_coeffs)."""
    from experiments.speech_commands.data import get_datasets, NUM_MFCC, NUM_FRAMES

    train_ds, _, _ = get_datasets(batch_size=4)

    for x, y in train_ds.take(1):
        assert x.shape == (4, NUM_FRAMES, NUM_MFCC), f"Got {x.shape}"
        assert y.shape == (4,)
        assert y.dtype == tf.int64


def test_labels_in_valid_range():
    """Labels are in range [0, num_classes)."""
    from experiments.speech_commands.data import get_datasets, NUM_CLASSES

    train_ds, _, _ = get_datasets(batch_size=32)

    for _, y in train_ds.take(5):
        assert tf.reduce_all(y >= 0)
        assert tf.reduce_all(y < NUM_CLASSES)


def test_enlarge_parameter_ignored():
    """get_datasets accepts enlarge parameter for DataLoader protocol compatibility."""
    from experiments.speech_commands.data import get_datasets, NUM_MFCC, NUM_FRAMES

    # Should work with enlarge=True (ignored for Speech Commands)
    train_ds, _, _ = get_datasets(batch_size=4, enlarge=True)

    for x, _ in train_ds.take(1):
        # Shape should be same regardless of enlarge parameter
        assert x.shape == (4, NUM_FRAMES, NUM_MFCC)

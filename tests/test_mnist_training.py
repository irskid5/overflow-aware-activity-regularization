"""Tests for experiments/mnist/training.py."""

import tensorflow as tf


def test_configure_environment_returns_strategy():
    from experiments.mnist.training import configure_environment

    strategy, dtype = configure_environment()
    assert isinstance(strategy, tf.distribute.Strategy)
    assert dtype == tf.float32


def test_configure_environment_returns_one_device_strategy():
    from experiments.mnist.training import configure_environment

    strategy, _ = configure_environment()
    assert isinstance(strategy, tf.distribute.OneDeviceStrategy)

"""Tests for experiments/mnist/model.py."""

import pytest
import tensorflow as tf

from oar.config import StepConfig
from experiments.mnist.model import get_model


# Helper: create a basic step config for tests
def _make_step_config(name: str = "test", **kwargs) -> StepConfig:
    """Create a minimal StepConfig for testing."""
    return StepConfig(name=name, **kwargs)


def test_get_model_accepts_step_config():
    step = _make_step_config("tanh_baseline")
    model = get_model(step)
    assert model is not None
    assert "MNIST_RNN" in model.name


def test_get_model_returns_keras_model():
    step = _make_step_config("test")
    model = get_model(step)
    assert isinstance(model, tf.keras.Model)


def test_get_model_has_correct_input_shape_28x28():
    step = _make_step_config("test", enlarge=False)
    model = get_model(step)
    assert model.input_shape == (None, 28, 28, 1)


def test_get_model_has_correct_output_shape():
    step = _make_step_config("test")
    model = get_model(step)
    assert model.output_shape == (None, 10)


def test_get_model_layer_names():
    step = _make_step_config("test")
    model = get_model(step)
    layer_names = [layer.name for layer in model.layers]
    assert "QRNN_0" in layer_names
    assert "QRNN_1" in layer_names
    assert "DENSE_0" in layer_names
    assert "DENSE_OUT" in layer_names


def test_get_model_has_correct_input_shape_128x128():
    step = _make_step_config("enlarged", enlarge=True)
    model = get_model(step)
    assert model.input_shape == (None, 128, 128, 1)

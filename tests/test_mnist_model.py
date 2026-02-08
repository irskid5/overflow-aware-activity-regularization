"""Tests for experiments/mnist/model.py."""

import pytest
import tensorflow as tf


@pytest.fixture
def default_options():
    """Default options for model tests."""
    return {
        "enlarge": False,
        "batch_size": 32,
        "quantize": False,
        "oar": {"oar_lambda": 1e-4, "omega": 6},
        "tᵢ": 0.7,
    }


@pytest.fixture
def default_layer_options():
    """Default layer options for model tests."""
    def make_layer_config(activation, oar_lambda=1e-4):
        return {
            "activation": activation,
            "oar": {"use": False, "oar_lambda": oar_lambda, "omega": 6},
            "s": 1.0,
            "τ": 0.0,
        }

    return {
        "INPUT": {"ternarize": False},
        "QRNN_0": make_layer_config(tf.keras.activations.tanh),
        "QRNN_1": make_layer_config(tf.keras.activations.tanh),
        "DENSE_0": make_layer_config(tf.keras.activations.tanh),
        "DENSE_OUT": make_layer_config(tf.keras.activations.softmax, oar_lambda=0.0),
    }


def test_get_model_returns_keras_model(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    model = get_model(default_options, default_layer_options)
    assert isinstance(model, tf.keras.Model)


def test_get_model_has_correct_input_shape_28x28(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    model = get_model(default_options, default_layer_options)
    assert model.input_shape == (None, 28, 28, 1)


def test_get_model_has_correct_output_shape(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    model = get_model(default_options, default_layer_options)
    assert model.output_shape == (None, 10)


def test_get_model_layer_names(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    model = get_model(default_options, default_layer_options)
    layer_names = [layer.name for layer in model.layers]
    assert "QRNN_0" in layer_names
    assert "QRNN_1" in layer_names
    assert "DENSE_0" in layer_names
    assert "DENSE_OUT" in layer_names


def test_get_model_has_correct_input_shape_128x128(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    default_options["enlarge"] = True
    model = get_model(default_options, default_layer_options)
    assert model.input_shape == (None, 128, 128, 1)

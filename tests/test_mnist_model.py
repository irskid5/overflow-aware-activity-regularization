"""Tests for experiments/mnist/model.py."""

import pytest
import tensorflow as tf


def test_get_model_accepts_step_config():
    from experiments.mnist.steps import STEP_1
    from experiments.mnist.model import get_model
    
    model = get_model(STEP_1)
    assert model is not None
    assert "MNIST_RNN" in model.name


def test_get_model_returns_keras_model():
    from experiments.mnist.steps import STEP_1
    from experiments.mnist.model import get_model

    model = get_model(STEP_1)
    assert isinstance(model, tf.keras.Model)


def test_get_model_has_correct_input_shape_28x28():
    from experiments.mnist.steps import STEP_1
    from experiments.mnist.model import get_model

    model = get_model(STEP_1)
    assert model.input_shape == (None, 28, 28, 1)


def test_get_model_has_correct_output_shape():
    from experiments.mnist.steps import STEP_1
    from experiments.mnist.model import get_model

    model = get_model(STEP_1)
    assert model.output_shape == (None, 10)


def test_get_model_layer_names():
    from experiments.mnist.steps import STEP_1
    from experiments.mnist.model import get_model

    model = get_model(STEP_1)
    layer_names = [layer.name for layer in model.layers]
    assert "QRNN_0" in layer_names
    assert "QRNN_1" in layer_names
    assert "DENSE_0" in layer_names
    assert "DENSE_OUT" in layer_names


def test_get_model_has_correct_input_shape_128x128():
    from oar.config import TrainingStepConfig
    from experiments.mnist.model import get_model

    step = TrainingStepConfig(name="enlarged", enlarge=True)
    model = get_model(step)
    assert model.input_shape == (None, 128, 128, 1)

"""Tests for experiments/mnist/config.py."""

import pytest
import tensorflow as tf


def test_make_oar_config_returns_expected_structure():
    """Test that _make_oar_config returns properly structured dict."""
    from experiments.mnist.config import _make_oar_config

    result = _make_oar_config(use=True, oar_lambda=1e-3, omega=8)

    assert result == {"use": True, "oar_lambda": 1e-3, "omega": 8}


def test_make_oar_config_with_defaults():
    """Test _make_oar_config with different values."""
    from experiments.mnist.config import _make_oar_config

    result = _make_oar_config(use=False, oar_lambda=0.0, omega=6)

    assert result["use"] is False
    assert result["oar_lambda"] == 0.0
    assert result["omega"] == 6


def test_mnist_options_has_required_keys():
    from experiments.mnist.config import MNIST_OPTIONS

    assert "enlarge" in MNIST_OPTIONS
    assert "epochs" in MNIST_OPTIONS
    assert "learning_rate" in MNIST_OPTIONS
    assert "batch_size" in MNIST_OPTIONS
    assert "t" in MNIST_OPTIONS
    assert "tᵢ" in MNIST_OPTIONS
    assert "s" in MNIST_OPTIONS
    assert "oar" in MNIST_OPTIONS
    assert "quantize" in MNIST_OPTIONS


def test_get_default_layer_options_returns_all_layers():
    from experiments.mnist.config import get_default_layer_options, MNIST_OPTIONS

    layer_options = get_default_layer_options(MNIST_OPTIONS)
    assert "INPUT" in layer_options
    assert "QRNN_0" in layer_options
    assert "QRNN_1" in layer_options
    assert "DENSE_0" in layer_options
    assert "DENSE_OUT" in layer_options


def test_get_default_layer_options_uses_omega():
    from experiments.mnist.config import get_default_layer_options

    options = {
        "oar": {"omega": 8, "oar_lambda": 1e-3},
    }
    layer_options = get_default_layer_options(options)
    assert layer_options["QRNN_0"]["oar"]["omega"] == 8
    assert layer_options["QRNN_0"]["oar"]["oar_lambda"] == 1e-3


def test_get_model_parameter_stats_returns_dict():
    from experiments.mnist.config import (
        get_model_parameter_stats,
        get_default_layer_options,
        MNIST_OPTIONS,
    )

    layer_options = get_default_layer_options(MNIST_OPTIONS)
    stats = get_model_parameter_stats(None, MNIST_OPTIONS, layer_options)
    assert "QRNN_0" in stats
    assert "QRNN_1" in stats
    assert "DENSE_0" in stats
    assert "DENSE_OUT" in stats


def test_perform_step_returns_checkpoint_path():
    from experiments.mnist.config import (
        perform_step_in_four_step_quant,
        MNIST_OPTIONS,
    )
    import copy

    # Use minimal epochs for test
    options = copy.deepcopy(MNIST_OPTIONS)
    options["epochs"] = 1

    # Step 1 should return a path string
    result = perform_step_in_four_step_quant(
        step=1, pretrained_weights=None, options=options
    )
    assert isinstance(result, str)
    assert "checkpoints" in result


def test_perform_step_accepts_run_dir_parameter():
    """Test that perform_step_in_four_step_quant accepts run_dir parameter."""
    from experiments.mnist.config import perform_step_in_four_step_quant
    import inspect

    sig = inspect.signature(perform_step_in_four_step_quant)
    param_names = list(sig.parameters.keys())

    assert "run_dir" in param_names, "Should accept run_dir parameter"

"""Tests for experiments/mnist/training.py."""

import os
import tempfile

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


def test_train_accepts_step_and_run_dir_parameters():
    """Test that train() accepts optional step and run_dir parameters."""
    from experiments.mnist.training import train
    import inspect

    sig = inspect.signature(train)
    param_names = list(sig.parameters.keys())

    assert "step" in param_names, "train() should accept 'step' parameter"
    assert "run_dir" in param_names, "train() should accept 'run_dir' parameter"


def test_train_creates_step_subdir_when_step_provided():
    """Test that train() creates step_N subdirectory when step is provided."""
    from experiments.mnist.training import train
    from experiments.mnist.config import MNIST_OPTIONS, get_default_layer_options
    import copy

    options = copy.deepcopy(MNIST_OPTIONS)
    options["epochs"] = 0  # Skip actual training
    layer_options = get_default_layer_options(options)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = tmpdir + "/test_run/"
        result = train(
            pretrained_weights=None,
            options=options,
            layer_options=layer_options,
            step=1,
            run_dir=run_dir,
        )

        # Should create step_1 subdirectory
        assert os.path.exists(os.path.join(run_dir, "step_1"))
        assert "step_1/checkpoints" in result


def test_train_saves_config_json_when_step_provided():
    """Test that train() saves config.json in step directory."""
    from experiments.mnist.training import train
    from experiments.mnist.config import MNIST_OPTIONS, get_default_layer_options
    import copy
    import json

    options = copy.deepcopy(MNIST_OPTIONS)
    options["epochs"] = 0  # Skip actual training
    layer_options = get_default_layer_options(options)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = tmpdir + "/test_run/"
        train(
            pretrained_weights=None,
            options=options,
            layer_options=layer_options,
            step=2,
            run_dir=run_dir,
        )

        config_path = os.path.join(run_dir, "step_2", "config.json")
        assert os.path.exists(config_path), "config.json should be created"

        with open(config_path) as f:
            saved_config = json.load(f)

        assert "options" in saved_config
        assert "layer_options" in saved_config
        assert "step" in saved_config
        assert saved_config["step"] == 2
        assert saved_config["options"]["batch_size"] == options["batch_size"]

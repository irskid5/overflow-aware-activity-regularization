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


def test_train_accepts_step_config():
    from experiments.mnist.training import train
    import inspect
    
    sig = inspect.signature(train)
    params = list(sig.parameters.keys())
    assert "step_config" in params
    assert "options" not in params
    assert "layer_options" not in params


def test_train_creates_step_subdir_when_step_provided():
    """Test that train() creates step_N subdirectory when step is provided."""
    from experiments.mnist.training import train
    from oar.config import TrainingStepConfig

    step_config = TrainingStepConfig(name="test", epochs=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = tmpdir + "/test_run/"
        result = train(
            step_config=step_config,
            step_number=1,
            pretrained_weights=None,
            run_dir=run_dir,
        )

        # Should create step_1 subdirectory
        assert os.path.exists(os.path.join(run_dir, "step_1"))
        assert "step_1/checkpoints" in result


def test_train_saves_config_json_when_step_provided():
    """Test that train() saves config.json in step directory."""
    from experiments.mnist.training import train
    from oar.config import TrainingStepConfig
    import json

    step_config = TrainingStepConfig(name="test_step", epochs=0, batch_size=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = tmpdir + "/test_run/"
        train(
            step_config=step_config,
            step_number=2,
            pretrained_weights=None,
            run_dir=run_dir,
        )

        config_path = os.path.join(run_dir, "step_2", "config.json")
        assert os.path.exists(config_path), "config.json should be created"

        with open(config_path) as f:
            saved_config = json.load(f)

        assert "step" in saved_config
        assert "step_config" in saved_config
        assert saved_config["step"] == 2
        assert saved_config["step_config"]["batch_size"] == 32
        assert saved_config["step_config"]["name"] == "test_step"

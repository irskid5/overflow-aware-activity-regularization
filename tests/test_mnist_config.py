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


def test_perform_four_step_quant_creates_shared_run_dir():
    """Test that all four steps share the same run directory."""
    from experiments.mnist.config import perform_four_step_quant, MNIST_OPTIONS
    import copy
    import os
    import tempfile
    from unittest.mock import patch

    # Patch RUNS_DIR to use temp directory
    import experiments.mnist.training as training_module

    with tempfile.TemporaryDirectory() as tmpdir:
        original_runs_dir = training_module.RUNS_DIR
        training_module.RUNS_DIR = tmpdir + "/"

        # Track which run_dir values were passed to each step
        step_run_dirs = []

        def mock_perform_step(step, pretrained_weights, options, run_dir=None):
            """Mock that records run_dir and creates step directories."""
            step_run_dirs.append((step, run_dir))
            # Create the step directory to simulate real behavior
            if run_dir is not None:
                step_dir = os.path.join(run_dir, f"step_{step}")
                os.makedirs(step_dir, exist_ok=True)
                ckpt_dir = os.path.join(step_dir, "checkpoints/")
                os.makedirs(ckpt_dir, exist_ok=True)
                return ckpt_dir
            return f"{tmpdir}/step_{step}/checkpoints/"

        try:
            options = copy.deepcopy(MNIST_OPTIONS)
            options["epochs"] = 0  # Doesn't matter with mock

            with patch(
                "experiments.mnist.config.perform_step_in_four_step_quant",
                side_effect=mock_perform_step,
            ):
                result = perform_four_step_quant(options)

            # All four steps should have been called
            assert len(step_run_dirs) == 4

            # All steps should have received the SAME run_dir
            run_dirs = [rd for _, rd in step_run_dirs]
            assert run_dirs[0] is not None, "run_dir should be provided"
            assert all(
                rd == run_dirs[0] for rd in run_dirs
            ), f"All steps should share same run_dir, got: {run_dirs}"

            # Verify the run_dir is in the expected location (tmpdir)
            assert run_dirs[0].startswith(tmpdir), f"run_dir should be in temp dir"

        finally:
            training_module.RUNS_DIR = original_runs_dir


def test_perform_four_step_quant_saves_initial_config():
    """Test that experiment-level config.json is saved in run directory."""
    from experiments.mnist.config import perform_four_step_quant, MNIST_OPTIONS
    import copy
    import json
    import os
    import tempfile
    from unittest.mock import patch

    import experiments.mnist.training as training_module

    with tempfile.TemporaryDirectory() as tmpdir:
        original_runs_dir = training_module.RUNS_DIR
        training_module.RUNS_DIR = tmpdir + "/"

        def mock_perform_step(step, pretrained_weights, options, run_dir=None):
            """Mock that creates step directories."""
            if run_dir is not None:
                step_dir = os.path.join(run_dir, f"step_{step}")
                os.makedirs(step_dir, exist_ok=True)
                ckpt_dir = os.path.join(step_dir, "checkpoints/")
                os.makedirs(ckpt_dir, exist_ok=True)
                return ckpt_dir
            return f"{tmpdir}/step_{step}/checkpoints/"

        try:
            options = copy.deepcopy(MNIST_OPTIONS)
            options["epochs"] = 0

            with patch(
                "experiments.mnist.config.perform_step_in_four_step_quant",
                side_effect=mock_perform_step,
            ):
                result = perform_four_step_quant(options)

            # Get the run directory
            run_dir = os.path.dirname(os.path.dirname(result.rstrip("/")))

            # Experiment-level config should exist
            config_path = os.path.join(run_dir, "config.json")
            assert os.path.exists(config_path), "Experiment config.json should exist"

            with open(config_path) as f:
                config = json.load(f)

            assert "initial_options" in config
            assert "started_at" in config

        finally:
            training_module.RUNS_DIR = original_runs_dir


def test_perform_four_step_quant_creates_output_log():
    """Test that experiment-level output.log is created."""
    from experiments.mnist.config import perform_four_step_quant, MNIST_OPTIONS
    import copy
    import os
    import tempfile
    from unittest.mock import patch

    import experiments.mnist.training as training_module

    with tempfile.TemporaryDirectory() as tmpdir:
        original_runs_dir = training_module.RUNS_DIR
        training_module.RUNS_DIR = tmpdir + "/"

        def mock_perform_step(step, pretrained_weights, options, run_dir=None):
            """Mock that creates step directories and prints step markers."""
            print(f"PERFORMING STEP {step}/4 FROM FOUR-STEP QUANTIZATION PROCESS")
            if run_dir is not None:
                step_dir = os.path.join(run_dir, f"step_{step}")
                os.makedirs(step_dir, exist_ok=True)
                ckpt_dir = os.path.join(step_dir, "checkpoints/")
                os.makedirs(ckpt_dir, exist_ok=True)
                return ckpt_dir
            return f"{tmpdir}/step_{step}/checkpoints/"

        try:
            options = copy.deepcopy(MNIST_OPTIONS)
            options["epochs"] = 0

            with patch(
                "experiments.mnist.config.perform_step_in_four_step_quant",
                side_effect=mock_perform_step,
            ):
                result = perform_four_step_quant(options)

            # Get the run directory
            run_dir = os.path.dirname(os.path.dirname(result.rstrip("/")))

            # Experiment-level log should exist
            log_path = os.path.join(run_dir, "output.log")
            assert os.path.exists(log_path), "Experiment output.log should exist"

            with open(log_path) as f:
                content = f.read()

            # Should contain output from all steps
            assert "STEP 1/4" in content
            assert "STEP 4/4" in content

        finally:
            training_module.RUNS_DIR = original_runs_dir

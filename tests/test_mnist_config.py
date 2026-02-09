"""Tests for experiments/mnist/config.py."""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from oar.config import TrainingStepConfig, LayerConfig
from experiments.mnist.steps import STEP_1, STATIC_STEPS


def test_get_model_parameter_stats_new_returns_dict_with_no_weights():
    """Test get_model_parameter_stats_new returns zeros when no pretrained weights."""
    from experiments.mnist.config import get_model_parameter_stats_new
    
    result = get_model_parameter_stats_new(None, STEP_1)
    
    assert isinstance(result, dict)
    assert "QRNN_0" in result
    assert "QRNN_1" in result
    assert "DENSE_0" in result
    assert "DENSE_OUT" in result
    assert all(v == 0.0 for v in result.values())


def test_get_model_parameter_stats_new_accepts_custom_layer_names():
    """Test get_model_parameter_stats_new accepts custom layer names."""
    from experiments.mnist.config import get_model_parameter_stats_new
    
    custom_layers = ["QRNN_0", "DENSE_0"]
    result = get_model_parameter_stats_new(None, STEP_1, layer_names=custom_layers)
    
    assert set(result.keys()) == set(custom_layers)


def test_perform_training_steps_raises_on_empty_steps():
    """Test that perform_training_steps raises ValueError for empty steps list."""
    from experiments.mnist.config import perform_training_steps
    
    with pytest.raises(ValueError, match="steps list cannot be empty"):
        perform_training_steps(steps=[])


def test_perform_training_steps_calls_train_for_each_step():
    """Test that perform_training_steps calls train for each step config."""
    from experiments.mnist.config import perform_training_steps
    import experiments.mnist.training as training_module
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_runs_dir = training_module.RUNS_DIR
        training_module.RUNS_DIR = tmpdir + "/"
        
        train_calls = []
        
        def mock_train(step_config, step_number, pretrained_weights, run_dir):
            train_calls.append({
                "step_config": step_config,
                "step_number": step_number,
                "pretrained_weights": pretrained_weights,
            })
            # Create checkpoint path - return a file path, not directory
            ckpt_dir = os.path.join(run_dir, f"step_{step_number}/checkpoints/")
            os.makedirs(ckpt_dir, exist_ok=True)
            # Return a file path
            ckpt_file = os.path.join(ckpt_dir, "weights.h5")
            with open(ckpt_file, "wb") as f:
                f.write(b"mock")
            return ckpt_file
        
        # Create minimal test steps
        test_steps = [
            TrainingStepConfig(name="test_step_1", epochs=1),
            TrainingStepConfig(name="test_step_2", epochs=1),
        ]
        
        try:
            with patch("experiments.mnist.training.train", side_effect=mock_train):
                # Pass None to skip step_4_factory entirely
                perform_training_steps(steps=test_steps, step_4_factory=None)
        finally:
            training_module.RUNS_DIR = original_runs_dir
        
        # Should have called train twice
        assert len(train_calls) == 2
        assert train_calls[0]["step_number"] == 1
        assert train_calls[1]["step_number"] == 2
        assert train_calls[0]["step_config"].name == "test_step_1"
        assert train_calls[1]["step_config"].name == "test_step_2"


def test_perform_training_steps_saves_experiment_config():
    """Test that perform_training_steps saves config.json with step configs."""
    from experiments.mnist.config import perform_training_steps
    import experiments.mnist.training as training_module
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_runs_dir = training_module.RUNS_DIR
        training_module.RUNS_DIR = tmpdir + "/"
        
        run_dir_captured = [None]
        
        def mock_train(step_config, step_number, pretrained_weights, run_dir):
            run_dir_captured[0] = run_dir
            ckpt_dir = os.path.join(run_dir, f"step_{step_number}/checkpoints/")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_file = os.path.join(ckpt_dir, "weights.h5")
            with open(ckpt_file, "wb") as f:
                f.write(b"mock")
            return ckpt_file
        
        test_steps = [TrainingStepConfig(name="config_test", epochs=1)]
        
        try:
            with patch("experiments.mnist.training.train", side_effect=mock_train):
                # Pass None to skip step_4_factory entirely
                perform_training_steps(steps=test_steps, step_4_factory=None)
            
            # Check config.json exists
            config_path = os.path.join(run_dir_captured[0], "config.json")
            assert os.path.exists(config_path)
            
            with open(config_path) as f:
                config = json.load(f)
            
            assert "started_at" in config
            assert "experiment_type" in config
            assert "steps" in config
            assert len(config["steps"]) == 1
            assert config["steps"][0]["name"] == "config_test"
            
        finally:
            training_module.RUNS_DIR = original_runs_dir


def test_perform_training_steps_uses_step_4_factory():
    """Test that perform_training_steps calls step_4_factory to create step 4."""
    from experiments.mnist.config import perform_training_steps
    import experiments.mnist.training as training_module
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_runs_dir = training_module.RUNS_DIR
        training_module.RUNS_DIR = tmpdir + "/"
        
        factory_called = [False]
        factory_tern_params = [None]
        
        def mock_factory(tern_params):
            factory_called[0] = True
            factory_tern_params[0] = tern_params
            return TrainingStepConfig(name="step_4_generated", epochs=1)
        
        def mock_train(step_config, step_number, pretrained_weights, run_dir):
            ckpt_dir = os.path.join(run_dir, f"step_{step_number}/checkpoints/")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_file = os.path.join(ckpt_dir, "weights.h5")
            with open(ckpt_file, "wb") as f:
                f.write(b"mock")
            return ckpt_file
        
        test_steps = [TrainingStepConfig(name="step_1", epochs=1)]
        
        try:
            with patch("experiments.mnist.training.train", side_effect=mock_train):
                with patch("experiments.mnist.config.get_model_parameter_stats_new", return_value={"QRNN_0": 0.1}):
                    perform_training_steps(steps=test_steps, step_4_factory=mock_factory)
            
            assert factory_called[0], "step_4_factory should have been called"
            assert factory_tern_params[0] == {"QRNN_0": 0.1}
            
        finally:
            training_module.RUNS_DIR = original_runs_dir

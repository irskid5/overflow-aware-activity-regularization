"""Tests for Phase 5 config dataclasses and Runner.

Tests validation logic, dacite integration, and Runner step validation.
"""

import pytest
from dacite import from_dict
from oar.config import (
    ActivationConfig, OARConfig, QuantizationConfig,
    LayerStepConfig, StepConfig, ExperimentConfig, DACITE_CONFIG,
)
from experiments.mnist.config import MNIST_CONFIG, MNIST_EXPERIMENT


# === ActivationConfig Tests ===

def test_activation_config_defaults():
    cfg = ActivationConfig()
    assert cfg.function == "tanh"
    assert cfg.gradient_scale is None
    assert cfg.omega is None

def test_activation_config_mod_sign_requires_omega():
    with pytest.raises(ValueError, match="omega required"):
        ActivationConfig(function="mod_sign")

def test_activation_config_mod_sign_with_omega():
    cfg = ActivationConfig(function="mod_sign", omega=6)
    assert cfg.function == "mod_sign"
    assert cfg.omega == 6

def test_activation_config_invalid_function():
    with pytest.raises(ValueError, match="must be one of"):
        ActivationConfig(function="invalid")

def test_activation_config_from_string_via_dacite():
    """Test string shorthand via DACITE_CONFIG type hook."""
    data = {"activation": "tanh"}
    cfg = from_dict(LayerStepConfig, data, config=DACITE_CONFIG)
    assert cfg.activation.function == "tanh"


# === OARConfig Tests ===

def test_oar_config_defaults():
    cfg = OARConfig()
    assert cfg.regularization_rate == 0.0
    assert cfg.omega == 6

def test_oar_config_negative_rate():
    with pytest.raises(ValueError, match="non-negative"):
        OARConfig(regularization_rate=-0.1)

def test_oar_config_invalid_omega():
    with pytest.raises(ValueError, match="omega must be >= 1"):
        OARConfig(omega=0)


# === QuantizationConfig Tests ===

def test_quantization_config_defaults():
    cfg = QuantizationConfig()
    assert cfg.ternarization_scale is None
    assert cfg.threshold is None
    assert cfg.oar is None

def test_quantization_config_negative_scale():
    with pytest.raises(ValueError, match="non-negative"):
        QuantizationConfig(ternarization_scale=-1.0)

def test_quantization_config_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        QuantizationConfig(threshold=-0.5)


# === StepConfig Tests ===

def test_step_config_defaults():
    cfg = StepConfig(name="test")
    assert cfg.epochs == 1000
    assert cfg.learning_rate == 1e-4
    assert cfg.batch_size == 512
    assert cfg.enlarge is False
    assert cfg.layers == {}

def test_step_config_negative_epochs():
    with pytest.raises(ValueError, match="non-negative"):
        StepConfig(name="test", epochs=-1)

def test_step_config_zero_learning_rate():
    with pytest.raises(ValueError, match="positive"):
        StepConfig(name="test", learning_rate=0)

def test_step_config_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        StepConfig(name="test", batch_size=0)


# === ExperimentConfig Tests ===

def test_experiment_from_dict():
    exp = from_dict(ExperimentConfig, MNIST_CONFIG, config=DACITE_CONFIG)
    assert exp.name == "mnist_four_step"
    assert exp.layer_names == ["INPUT", "QRNN_0", "QRNN_1", "DENSE_0", "DENSE_OUT"]
    assert 1 in exp.steps
    assert exp.steps[1].name == "tanh_baseline"

def test_experiment_step_4_has_oar():
    assert 4 in MNIST_EXPERIMENT.steps
    qrnn0 = MNIST_EXPERIMENT.steps[4].layers["QRNN_0"]
    assert qrnn0.quantization.oar is not None
    assert qrnn0.quantization.oar.regularization_rate == 1e-4

def test_experiment_input_layer_in_step_3():
    assert "INPUT" in MNIST_EXPERIMENT.steps[3].layers
    input_cfg = MNIST_EXPERIMENT.steps[3].layers["INPUT"]
    assert input_cfg.quantization.ternarization_scale == 0.7


# === Runner Tests ===

def test_runner_validates_step_range():
    from oar.runner import Runner
    
    runner = Runner(
        experiment=MNIST_EXPERIMENT,
        model_factory=lambda cfg, **kw: None,
        data_loader=lambda **kw: (None, None, None),
    )
    
    with pytest.raises(ValueError, match="start_step.*> end_step"):
        runner.run(start_step=4, end_step=2)

def test_runner_validates_missing_step():
    from oar.runner import Runner
    
    # Create experiment with only step 1 and 3 (missing step 2)
    config = {
        "name": "test",
        "layer_names": [],
        "steps": {
            1: {"name": "step1"},
            3: {"name": "step3"},
        }
    }
    exp = from_dict(ExperimentConfig, config, config=DACITE_CONFIG)
    
    runner = Runner(
        experiment=exp,
        model_factory=lambda cfg, **kw: None,
        data_loader=lambda **kw: (None, None, None),
    )
    
    with pytest.raises(ValueError, match="Step 2 not defined"):
        runner.run(start_step=1, end_step=3)

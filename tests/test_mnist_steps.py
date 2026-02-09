"""Tests for experiments/mnist/steps.py."""

import pytest


def test_static_steps_count():
    from experiments.mnist.steps import STATIC_STEPS
    assert len(STATIC_STEPS) == 3


def test_step_1_defaults():
    from experiments.mnist.steps import STEP_1
    assert STEP_1.name == "tanh_baseline"
    assert STEP_1.epochs == 100
    assert STEP_1.layers == {}


def test_step_2_gradient_scaling():
    from experiments.mnist.steps import STEP_2
    assert STEP_2.name == "sign_with_gradient_scaling"
    assert "QRNN_0" in STEP_2.layers
    assert STEP_2.layers["QRNN_0"].gradient_scale == 4.0
    assert STEP_2.layers["QRNN_0"].activation == "sign_ste_tanh"


def test_step_3_input_quantization():
    from experiments.mnist.steps import STEP_3
    assert STEP_3.input_config.quantize_threshold == 0.7


def test_create_step_4():
    from experiments.mnist.steps import create_step_4
    
    tern_params = {"QRNN_0": 0.1, "QRNN_1": 0.1, "DENSE_0": 0.1, "DENSE_OUT": 0.1}
    step = create_step_4(tern_params, t=1.5)
    
    assert step.name == "full_quantization_with_oar"
    assert step.learning_rate == 1e-5
    assert step.layers["QRNN_0"].quantize_threshold == pytest.approx(0.15)  # 1.5 * 0.1
    assert step.layers["QRNN_0"].oar_lambda == 1e-4


def test_get_default_layer_config():
    from experiments.mnist.steps import get_default_layer_config
    
    # Regular layer defaults to tanh
    cfg = get_default_layer_config("QRNN_0")
    assert cfg.activation == "tanh"
    
    # DENSE_OUT defaults to softmax
    cfg = get_default_layer_config("DENSE_OUT")
    assert cfg.activation == "softmax"


def test_compute_thresholds():
    from experiments.mnist.steps import compute_thresholds
    
    tern_params = {"QRNN_0": 0.1, "DENSE_0": 0.2}
    thresholds = compute_thresholds(tern_params, t=2.0)
    
    assert thresholds["QRNN_0"] == pytest.approx(0.2)
    assert thresholds["DENSE_0"] == pytest.approx(0.4)


def test_create_step_4_missing_layers():
    from experiments.mnist.steps import create_step_4
    
    # Only provide some layers - missing keys should default to 0
    tern_params = {"QRNN_0": 0.1}
    step = create_step_4(tern_params)
    
    assert step.layers["QRNN_0"].quantize_threshold == pytest.approx(0.15)
    assert step.layers["QRNN_1"].quantize_threshold == 0  # Missing -> 0

"""Tests for PTB experiment configuration."""
import pytest


def test_ptb_experiment_is_experiment_config():
    """PTB_EXPERIMENT is an ExperimentConfig."""
    from experiments.ptb.config import PTB_EXPERIMENT
    from oar.config import ExperimentConfig

    assert isinstance(PTB_EXPERIMENT, ExperimentConfig)


def test_ptb_has_four_steps():
    """PTB experiment has 4 training steps."""
    from experiments.ptb.config import PTB_EXPERIMENT

    assert len(PTB_EXPERIMENT.steps) == 4
    assert set(PTB_EXPERIMENT.steps.keys()) == {1, 2, 3, 4}


def test_step_1_is_tanh_baseline():
    """Step 1 uses tanh activation (baseline)."""
    from experiments.ptb.config import PTB_EXPERIMENT

    step1 = PTB_EXPERIMENT.steps[1]
    assert step1.name == "tanh_baseline"
    assert step1.layers["QRNN_0"].activation.function == "tanh"
    assert step1.layers["QRNN_1"].activation.function == "tanh"


def test_step_2_uses_sign_activation():
    """Step 2 uses sign_ste_tanh activation."""
    from experiments.ptb.config import PTB_EXPERIMENT

    step2 = PTB_EXPERIMENT.steps[2]
    assert step2.name == "sign_activation"
    assert step2.layers["QRNN_0"].activation.function == "sign_ste_tanh"


def test_step_3_has_input_quantization():
    """Step 3 ternarizes embedding output."""
    from experiments.ptb.config import PTB_EXPERIMENT

    step3 = PTB_EXPERIMENT.steps[3]
    assert step3.name == "input_quantization"
    assert step3.layers["INPUT"].quantization.ternarization_scale is not None


def test_step_4_has_oar():
    """Step 4 enables OAR regularization."""
    from experiments.ptb.config import PTB_EXPERIMENT

    step4 = PTB_EXPERIMENT.steps[4]
    assert step4.name == "full_quantization"
    assert step4.layers["QRNN_0"].quantization.oar is not None
    assert step4.layers["QRNN_0"].quantization.oar.regularization_rate > 0


def test_layer_names_exclude_dense_0():
    """PTB doesn't use DENSE_0 (no hidden dense layer)."""
    from experiments.ptb.config import PTB_EXPERIMENT

    assert "DENSE_0" not in PTB_EXPERIMENT.layer_names
    assert "DENSE_OUT" in PTB_EXPERIMENT.layer_names


def test_ptb_constants_defined():
    """PTB-specific constants are defined."""
    from experiments.ptb.config import BATCH_SIZE, NUM_STEPS, EMBED_DIM

    assert BATCH_SIZE == 20
    assert NUM_STEPS == 35
    assert EMBED_DIM == 128

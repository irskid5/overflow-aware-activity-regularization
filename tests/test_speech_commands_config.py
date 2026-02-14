"""Tests for Speech Commands experiment config."""


def test_config_has_four_steps():
    """Config defines all four quantization steps."""
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    
    assert len(SPEECH_COMMANDS_EXPERIMENT.steps) == 4
    assert set(SPEECH_COMMANDS_EXPERIMENT.steps.keys()) == {1, 2, 3, 4}


def test_config_layer_names():
    """Config defines expected layer names."""
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    
    expected = ["INPUT", "QRNN_0", "QRNN_1", "DENSE_0", "DENSE_OUT"]
    assert SPEECH_COMMANDS_EXPERIMENT.layer_names == expected


def test_step4_has_oar():
    """Step 4 enables OAR regularization on RNN layers."""
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    
    step4 = SPEECH_COMMANDS_EXPERIMENT.steps[4]
    assert step4.layers["QRNN_0"].quantization.oar is not None
    assert step4.layers["QRNN_1"].quantization.oar is not None

"""Tests for Speech Commands training utilities."""


def test_run_speech_commands_callable():
    """run_speech_commands function exists and is callable."""
    from experiments.speech_commands.training import run_speech_commands
    
    assert callable(run_speech_commands)


def test_data_loader_protocol():
    """make_data_loader returns DataLoader-compatible callable."""
    from experiments.speech_commands.training import make_data_loader
    import tensorflow as tf
    
    loader = make_data_loader()
    train, val, test = loader(batch_size=32, enlarge=False)
    
    # Check datasets are tf.data.Dataset (DataLoader protocol)
    assert isinstance(train, tf.data.Dataset)
    assert isinstance(val, tf.data.Dataset)
    assert isinstance(test, tf.data.Dataset)

"""Tests for Speech Commands model."""
import tensorflow as tf


def test_model_builds():
    """Model builds with step 1 config."""
    from experiments.speech_commands.model import get_model
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    
    step_config = SPEECH_COMMANDS_EXPERIMENT.steps[1]
    model = get_model(step_config)
    
    assert isinstance(model, tf.keras.Model)


def test_model_output_shape():
    """Model outputs (batch, num_classes)."""
    from experiments.speech_commands.model import get_model
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    from experiments.speech_commands.data import NUM_CLASSES, NUM_FRAMES, NUM_MFCC
    
    step_config = SPEECH_COMMANDS_EXPERIMENT.steps[1]
    model = get_model(step_config)
    
    # Create dummy input
    x = tf.zeros((4, NUM_FRAMES, NUM_MFCC))
    y = model(x, training=False)
    
    assert y.shape == (4, NUM_CLASSES)


def test_model_trainable():
    """Model can compute gradients."""
    from experiments.speech_commands.model import get_model
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    from experiments.speech_commands.data import NUM_FRAMES, NUM_MFCC
    
    step_config = SPEECH_COMMANDS_EXPERIMENT.steps[1]
    model = get_model(step_config)
    
    x = tf.random.normal((4, NUM_FRAMES, NUM_MFCC))
    y = tf.constant([0, 1, 2, 3])
    
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
    
    grads = tape.gradient(loss, model.trainable_variables)
    assert any(g is not None for g in grads)


def test_model_has_correct_input_shape():
    """Model has correct input shape."""
    from experiments.speech_commands.model import get_model
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    from experiments.speech_commands.data import NUM_FRAMES, NUM_MFCC
    
    step_config = SPEECH_COMMANDS_EXPERIMENT.steps[1]
    model = get_model(step_config)
    
    assert model.input_shape == (None, NUM_FRAMES, NUM_MFCC)


def test_model_layer_names():
    """Model has expected layer names."""
    from experiments.speech_commands.model import get_model
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    
    step_config = SPEECH_COMMANDS_EXPERIMENT.steps[1]
    model = get_model(step_config)
    
    layer_names = [layer.name for layer in model.layers]
    for expected in ["QRNN_0", "QRNN_1", "DENSE_0", "DENSE_OUT"]:
        assert any(expected in name for name in layer_names), f"Missing {expected}"


def test_model_name():
    """Model has correct name."""
    from experiments.speech_commands.model import get_model
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    
    step_config = SPEECH_COMMANDS_EXPERIMENT.steps[1]
    model = get_model(step_config)
    
    assert "SPEECH_COMMANDS_RNN" in model.name

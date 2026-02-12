"""Tests for PTB model builder."""
import pytest
import tensorflow as tf


def test_get_model_returns_oar_model():
    """_get_model returns an OARModel."""
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT
    from oar import OARModel

    step_config = PTB_EXPERIMENT.steps[1]
    model = _get_model(step_config, vocab_size=100)

    assert isinstance(model, OARModel)


def test_model_has_correct_input_shape():
    """Model input shape matches (batch_size, num_steps)."""
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT, BATCH_SIZE, NUM_STEPS

    step_config = PTB_EXPERIMENT.steps[1]
    model = _get_model(step_config, vocab_size=100)

    # Stateful model has batch_size in input shape
    assert model.input_shape == (BATCH_SIZE, NUM_STEPS)


def test_model_has_correct_output_shape():
    """Model output shape is (batch_size, num_steps, vocab_size)."""
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT, BATCH_SIZE, NUM_STEPS

    vocab_size = 100
    step_config = PTB_EXPERIMENT.steps[1]
    model = _get_model(step_config, vocab_size=vocab_size)

    assert model.output_shape == (BATCH_SIZE, NUM_STEPS, vocab_size)


def test_model_is_stateful():
    """Model uses stateful RNN layers."""
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT
    from oar.layers import QRNNWithOAR

    step_config = PTB_EXPERIMENT.steps[1]
    model = _get_model(step_config, vocab_size=100)

    # Find QRNN layers and check stateful
    rnn_layers = [l for l in model.layers if isinstance(l, QRNNWithOAR)]
    assert len(rnn_layers) >= 2  # QRNN_0, QRNN_1
    assert all(l.stateful for l in rnn_layers)


def test_model_can_reset_states():
    """Model has reset_states method."""
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT

    step_config = PTB_EXPERIMENT.steps[1]
    model = _get_model(step_config, vocab_size=100)

    # Should not raise
    model.reset_states()


def test_model_forward_pass():
    """Model can do forward pass with correct shapes."""
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT, BATCH_SIZE, NUM_STEPS

    vocab_size = 100
    step_config = PTB_EXPERIMENT.steps[1]
    model = _get_model(step_config, vocab_size=vocab_size)

    # Create input batch
    x = tf.random.uniform((BATCH_SIZE, NUM_STEPS), maxval=vocab_size, dtype=tf.int32)
    y = model(x)

    assert y.shape == (BATCH_SIZE, NUM_STEPS, vocab_size)


def test_step_4_model_has_oar_layers():
    """Step 4 model uses QRNNWithOAR with OAR enabled."""
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT
    from oar.layers import QRNNWithOAR

    step_config = PTB_EXPERIMENT.steps[4]
    model = _get_model(step_config, vocab_size=100)

    qrnn_layers = [l for l in model.layers if isinstance(l, QRNNWithOAR)]
    assert len(qrnn_layers) >= 1


def test_model_loads_pretrained_weights(tmp_path):
    """_get_model can load pretrained weights."""
    from experiments.ptb.model import _get_model
    from experiments.ptb.config import PTB_EXPERIMENT

    vocab_size = 100
    step_config = PTB_EXPERIMENT.steps[1]

    # Create and save model
    model1 = _get_model(step_config, vocab_size=vocab_size)
    weights_path = tmp_path / "weights.h5"
    model1.save_weights(str(weights_path))

    # Load into new model
    model2 = _get_model(step_config, vocab_size=vocab_size, pretrained_weights=str(weights_path))

    # Weights should match
    for w1, w2 in zip(model1.get_weights(), model2.get_weights()):
        assert (w1 == w2).all()


def test_make_model_factory_returns_callable():
    """make_model_factory returns ModelFactory-compatible callable."""
    from experiments.ptb.model import make_model_factory
    from experiments.ptb.config import PTB_EXPERIMENT

    vocab_size = 100
    model_factory = make_model_factory(vocab_size)

    # Should accept (step_config, pretrained_weights=None)
    step_config = PTB_EXPERIMENT.steps[1]
    model = model_factory(step_config)

    assert model is not None


def test_make_model_factory_with_pretrained_weights(tmp_path):
    """make_model_factory works with pretrained_weights."""
    from experiments.ptb.model import make_model_factory, _get_model
    from experiments.ptb.config import PTB_EXPERIMENT

    vocab_size = 100
    step_config = PTB_EXPERIMENT.steps[1]

    # Save weights
    model1 = _get_model(step_config, vocab_size=vocab_size)
    weights_path = tmp_path / "weights.h5"
    model1.save_weights(str(weights_path))

    # Load via factory
    model_factory = make_model_factory(vocab_size)
    model2 = model_factory(step_config, pretrained_weights=str(weights_path))

    assert model2 is not None

"""Tests for resolve_activation utility."""
import tensorflow as tf


def test_resolve_tanh():
    """resolve_activation('tanh') returns tanh function."""
    from oar import resolve_activation
    from oar.config import ActivationConfig
    
    config = ActivationConfig(function="tanh")
    fn = resolve_activation(config)
    
    x = tf.constant([0.0, 1.0, -1.0])
    expected = tf.tanh(x)
    tf.debugging.assert_near(fn(x), expected)


def test_resolve_sign_ste_tanh():
    """resolve_activation('sign_ste_tanh') returns sign_ste_tanh function."""
    from oar import resolve_activation, sign_ste_tanh
    from oar.config import ActivationConfig
    
    config = ActivationConfig(function="sign_ste_tanh")
    fn = resolve_activation(config)
    
    assert fn is sign_ste_tanh


def test_resolve_mod_sign():
    """resolve_activation('mod_sign') returns partial with omega."""
    from oar import resolve_activation
    from oar.config import ActivationConfig
    
    config = ActivationConfig(function="mod_sign", omega=6)
    fn = resolve_activation(config)
    
    # Should be callable
    x = tf.constant([0.0, 1.0])
    result = fn(x)
    assert result.shape == x.shape


def test_resolve_softmax():
    """resolve_activation('softmax') returns softmax function."""
    from oar import resolve_activation
    from oar.config import ActivationConfig
    
    config = ActivationConfig(function="softmax")
    fn = resolve_activation(config)
    
    assert fn is tf.keras.activations.softmax


def test_resolve_unknown_raises():
    """resolve_activation raises ValueError for unknown function."""
    import pytest
    from oar import resolve_activation
    from oar.config import ActivationConfig
    
    # Note: ActivationConfig validates in __post_init__, so we need to bypass
    config = ActivationConfig(function="tanh")
    config.function = "unknown_func"  # Bypass validation
    
    with pytest.raises(ValueError, match="Unknown activation"):
        resolve_activation(config)

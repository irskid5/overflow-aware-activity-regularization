"""Tests for oar/config.py dataclasses."""

import pytest


class TestLayerConfig:
    def test_default_values(self):
        from oar.config import LayerConfig
        config = LayerConfig()
        assert config.activation == "tanh"
        assert config.gradient_scale == 1.0
        assert config.oar_lambda is None
        assert config.omega == 6
        assert config.quantize_threshold is None
    
    def test_validates_activation(self):
        from oar.config import LayerConfig
        with pytest.raises(ValueError, match="activation must be one of"):
            LayerConfig(activation="invalid")
    
    def test_validates_gradient_scale(self):
        from oar.config import LayerConfig
        with pytest.raises(ValueError, match="gradient_scale must be positive"):
            LayerConfig(gradient_scale=0)
    
    def test_validates_oar_lambda(self):
        from oar.config import LayerConfig
        with pytest.raises(ValueError, match="oar_lambda must be non-negative"):
            LayerConfig(oar_lambda=-1e-4)
    
    def test_allows_none_oar_lambda(self):
        from oar.config import LayerConfig
        config = LayerConfig(oar_lambda=None)
        assert config.oar_lambda is None

    def test_validates_omega(self):
        from oar.config import LayerConfig
        with pytest.raises(ValueError, match="omega must be positive"):
            LayerConfig(omega=0)

    def test_validates_quantize_threshold(self):
        from oar.config import LayerConfig
        with pytest.raises(ValueError, match="quantize_threshold must be non-negative"):
            LayerConfig(quantize_threshold=-0.5)


class TestInputConfig:
    def test_default_none(self):
        from oar.config import InputConfig
        config = InputConfig()
        assert config.quantize_threshold is None
    
    def test_validates_threshold(self):
        from oar.config import InputConfig
        with pytest.raises(ValueError, match="quantize_threshold must be non-negative"):
            InputConfig(quantize_threshold=-0.1)


class TestTrainingStepConfig:
    def test_requires_name(self):
        from oar.config import TrainingStepConfig
        config = TrainingStepConfig(name="test")
        assert config.name == "test"
        assert config.epochs == 100
        assert config.learning_rate == 1e-4
    
    def test_validates_epochs(self):
        from oar.config import TrainingStepConfig
        with pytest.raises(ValueError, match="epochs must be non-negative"):
            TrainingStepConfig(name="test", epochs=-1)
    
    def test_allows_zero_epochs(self):
        from oar.config import TrainingStepConfig
        config = TrainingStepConfig(name="test", epochs=0)
        assert config.epochs == 0

    def test_validates_learning_rate(self):
        from oar.config import TrainingStepConfig
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingStepConfig(name="test", learning_rate=0)

    def test_validates_batch_size(self):
        from oar.config import TrainingStepConfig
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingStepConfig(name="test", batch_size=0)


class TestResolveActivation:
    def test_tanh(self):
        from oar.config import resolve_activation
        import tensorflow as tf
        result = resolve_activation("tanh")
        assert result == tf.keras.activations.tanh
    
    def test_sign_ste_tanh(self):
        from oar.config import resolve_activation
        result = resolve_activation("sign_ste_tanh")
        assert callable(result)
    
    def test_mod_sign(self):
        from oar.config import resolve_activation
        result = resolve_activation("mod_sign", omega=6)
        assert callable(result)
    
    def test_softmax(self):
        from oar.config import resolve_activation
        import tensorflow as tf
        result = resolve_activation("softmax")
        assert result == tf.keras.activations.softmax
    
    def test_invalid(self):
        from oar.config import resolve_activation
        with pytest.raises(ValueError, match="Unknown activation"):
            resolve_activation("invalid")

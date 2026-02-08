import tensorflow as tf

# Test imports from new location
from oar.regularizers import OAR1, OAR2, oar_penalty_fn, compute_oar_metric


def test_oar1_get_config_returns_all_params():
    layer = OAR1(oar_lambda=1e-3, k=64, a=2.0, name="test_oar1")
    config = layer.get_config()
    assert config["oar_lambda"] == 1e-3
    assert config["k"] == 64
    assert config["a"] == 2.0


def test_oar2_get_config_returns_all_params():
    layer = OAR2(oar_lambda=5e-4, k=128, a=1.5, name="test_oar2")
    config = layer.get_config()
    assert config["oar_lambda"] == 5e-4
    assert config["k"] == 128
    assert config["a"] == 1.5


def test_oar1_serialization_roundtrip():
    original = OAR1(oar_lambda=1e-3, k=64, a=2.0, name="test")
    config = original.get_config()
    restored = OAR1.from_config(config)
    assert restored.oar_lambda == original.oar_lambda
    assert restored.k == original.k
    assert restored.a == original.a


def test_oar2_serialization_roundtrip():
    original = OAR2(oar_lambda=5e-4, k=128, a=1.5, name="test")
    config = original.get_config()
    restored = OAR2.from_config(config)
    assert restored.oar_lambda == original.oar_lambda
    assert restored.k == original.k
    assert restored.a == original.a


def test_oar_penalty_fn_unchanged():
    x = tf.constant([0.0, 0.0, 0.0])
    out = oar_penalty_fn(x, k=8, a=1.0)
    tf.debugging.assert_equal(out, tf.zeros_like(out))


def test_compute_oar_metric_unchanged():
    x = tf.constant([[0.0, 0.0]])
    out = compute_oar_metric(x, k=8, a=1.0)
    tf.debugging.assert_near(out, tf.constant([1.0]))

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


from oar.activations import sign_ste_tanh, mod_sign, TrackedActivation


def test_sign_ste_tanh_unchanged():
    x = tf.constant([-2.0, 0.0, 2.0])
    out = sign_ste_tanh(x)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 1.0, 1.0]))


def test_mod_sign_unchanged():
    x = tf.constant([-1.0, 1.0, 5.0])
    out = mod_sign(x, num_bits=3)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 1.0, -1.0]))


def test_tracked_activation_get_config_fixed():
    """Test that TrackedActivation.get_config() no longer references alpha_init."""
    layer = TrackedActivation(activation=tf.nn.relu, name="test_act")
    config = layer.get_config()
    # Should have activation and name, NOT alpha_init
    assert "activation" in config or "name" in config
    assert "alpha_init" not in config


def test_tracked_activation_serialization_roundtrip():
    original = TrackedActivation(activation="relu", name="test")
    config = original.get_config()
    # Should not raise
    restored = TrackedActivation.from_config(config)
    assert restored is not None


def test_tracked_activation_applies_activation():
    layer = TrackedActivation(activation=tf.nn.relu)
    x = tf.constant([-1.0, 2.0])
    out = layer(x)
    tf.debugging.assert_equal(out, tf.constant([0.0, 2.0]))


from oar.quantizers import TernarizationWithThreshold, ternarize_tensor_with_threshold


def test_ternarize_tensor_with_threshold_unchanged():
    x = tf.constant([-2.0, -0.1, 0.1, 2.0])
    out = ternarize_tensor_with_threshold(x, theta=0.5)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 0.0, 0.0, 1.0]))


def test_ternarization_get_config_returns_all_params():
    """Test that get_config returns ALL constructor parameters (bug fix)."""
    quantizer = TernarizationWithThreshold(
        threshold=0.5,
        qnoise_factor=0.8,
        var_name="test_var",
        use_ste=False,
        use_variables=True,
        name="test_quant",
    )
    config = quantizer.get_config()
    assert config["threshold"] == 0.5
    assert config["qnoise_factor"] == 0.8
    assert config["var_name"] == "test_var"
    assert config["use_ste"] == False
    assert config["use_variables"] == True


def test_ternarization_serialization_roundtrip():
    original = TernarizationWithThreshold(
        threshold=0.3,
        qnoise_factor=0.9,
        use_ste=True,
    )
    config = original.get_config()
    restored = TernarizationWithThreshold.from_config(config)
    assert restored.threshold == original.threshold
    assert restored.qnoise_factor == original.qnoise_factor
    assert restored.use_ste == original.use_ste


def test_ternarization_call_produces_ternary():
    quantizer = TernarizationWithThreshold(threshold=0.5)
    x = tf.constant([-2.0, -0.1, 0.1, 2.0])
    out = quantizer(x)
    # With STE, output should be ternary values
    unique_vals = tf.unique(out)[0]
    # Should only have values in {-1, 0, 1}
    for val in unique_vals.numpy():
        assert val in [-1.0, 0.0, 1.0]


import tempfile
from oar.callbacks import (
    ReservoirHistogramCallback,
    _reservoir_update,
    reset_stat_weights,
    RESERVOIR_UPDATE_EVERY,
)


def test_reservoir_update_from_oar_callbacks():
    reservoir = tf.zeros([10], dtype=tf.float32)
    count = tf.constant(0, dtype=tf.int64)
    new_vals = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)

    updated, updated_count = _reservoir_update(
        reservoir=reservoir,
        count=count,
        new_values=new_vals,
        reservoir_size=10,
        seed=123,
    )

    tf.debugging.assert_equal(updated_count, tf.constant(4, dtype=tf.int64))


def test_reservoir_histogram_callback_from_oar():
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = ReservoirHistogramCallback(log_dir=tmpdir)
        assert cb is not None
        assert cb.log_dir == tmpdir


def test_reservoir_update_every_constant():
    assert RESERVOIR_UPDATE_EVERY == 20

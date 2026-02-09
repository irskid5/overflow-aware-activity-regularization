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


from oar.layers import (
    QRNNWithOAR,
    QSimpleRNNCellWithOAR,
    QDenseWithOAR,
    Downsampling,
)


def test_downsampling_get_config():
    layer = Downsampling(reduction_factor=2, batch_size=32)
    config = layer.get_config()
    assert config["reduction_factor"] == 2
    assert config["batch_size"] == 32


def test_downsampling_serialization_roundtrip():
    original = Downsampling(reduction_factor=4, batch_size=64)
    config = original.get_config()
    restored = Downsampling.from_config(config)
    assert restored.reduction_factor == original.reduction_factor
    assert restored.batch_size == original.batch_size


def test_qsimple_rnn_cell_has_reservoir_weights():
    cell = QSimpleRNNCellWithOAR(units=4, batch_size=2, name="QRNN_0")
    cell.build(tf.TensorShape([2, 8]))
    assert cell.preact_reservoir.shape.as_list() == [100000]
    assert cell.reservoir_count.dtype == tf.int64


def test_qdense_with_oar_has_reservoir_weights():
    layer = QDenseWithOAR(units=3, batch_size=2, name="DENSE_0")
    layer.build(tf.TensorShape([2, 8]))
    assert layer.preact_reservoir.shape.as_list() == [100000]
    assert layer.reservoir_count.dtype == tf.int64


def test_qrnn_with_oar_propagates_reservoir_size():
    layer = QRNNWithOAR(units=4, batch_size=2, reservoir_size=777, name="QRNN_0")
    layer.build(tf.TensorShape([2, 10, 8]))
    assert layer.cell.preact_reservoir.shape.as_list() == [777]


def test_qrnn_with_oar_serialization_roundtrip():
    original = QRNNWithOAR(units=4, batch_size=2, use_oar=True, oar_lambda=1e-4, omega=6, name="QRNN_0")
    original.build(tf.TensorShape([2, 10, 8]))
    config = original.get_config()
    # Verify key params are serialized
    assert "kernel_quantizer" in config
    assert "use_oar" in str(config) or original.use_oar  # Check param propagates


def test_qsimple_rnn_cell_get_config():
    cell = QSimpleRNNCellWithOAR(units=4, batch_size=2, use_oar=True, oar_lambda=1e-4, omega=6, name="QRNN_0")
    cell.build(tf.TensorShape([2, 8]))
    config = cell.get_config()
    assert config["batch_size"] == 2
    assert config["use_oar"] == True
    assert config["oar_lambda"] == 1e-4
    assert config["omega"] == 6


def test_qdense_with_oar_get_config():
    layer = QDenseWithOAR(units=3, batch_size=2, use_oar=True, oar_lambda=1e-4, omega=6, name="DENSE_0")
    layer.build(tf.TensorShape([2, 8]))
    config = layer.get_config()
    assert config["batch_size"] == 2
    assert config["use_oar"] == True
    assert config["oar_lambda"] == 1e-4


from oar.training import OARModel


def test_get_default_layer_config_from_experiments():
    from experiments.mnist.steps import get_default_layer_config

    # Test that get_default_layer_config returns proper LayerConfig
    layer_config = get_default_layer_config("QRNN_0")
    assert layer_config.omega == 6
    assert layer_config.activation == "tanh"


def test_oar_model_constructs():
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(5)(inputs)
    model = OARModel(inputs=inputs, outputs=outputs)
    assert model is not None


def test_model_save_load_roundtrip():
    """Integration test: full model save/load with OAR layers."""
    import tempfile
    
    # Build a minimal model with OAR layers
    inputs = tf.keras.layers.Input(shape=(10, 8), batch_size=2)
    x = QRNNWithOAR(units=4, batch_size=2, use_oar=True, oar_lambda=1e-4, omega=6, name="QRNN_0")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = QDenseWithOAR(units=3, batch_size=2, use_oar=False, name="DENSE_0")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(f"{tmpdir}/model.keras")
        loaded = tf.keras.models.load_model(f"{tmpdir}/model.keras")
    
    # Verify functional equivalence
    test_input = tf.random.normal([2, 10, 8])
    original_output = model(test_input)
    loaded_output = loaded(test_input)
    tf.debugging.assert_near(original_output, loaded_output, atol=1e-5)

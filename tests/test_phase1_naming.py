import tensorflow as tf

from oar import (
    compute_oar_metric,
    oar_penalty_fn,
    sign_ste_tanh,
    mod_sign,
    TrackedActivation,
)


def test_sign_ste_tanh_outputs_sign():
    x = tf.constant([-2.0, 0.0, 2.0])
    out = sign_ste_tanh(x)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 1.0, 1.0]))


def test_mod_sign_outputs_signed_mod():
    x = tf.constant([-1.0, 1.0, 5.0])
    out = mod_sign(x, num_bits=3)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 1.0, -1.0]))


def test_oar_penalty_is_zero_at_origin():
    x = tf.constant([0.0, 0.0, 0.0])
    out = oar_penalty_fn(x, k=8, a=1.0)
    tf.debugging.assert_equal(out, tf.zeros_like(out))


def test_compute_oar_metric_is_one_at_origin():
    x = tf.constant([[0.0, 0.0]])
    out = compute_oar_metric(x, k=8, a=1.0)
    tf.debugging.assert_near(out, tf.constant([1.0]))


def test_tracked_activation_applies_activation():
    layer = TrackedActivation(activation=tf.nn.relu)
    x = tf.constant([-1.0, 2.0])
    out = layer(x)
    tf.debugging.assert_equal(out, tf.constant([0.0, 2.0]))


def test_default_layer_options_use_omega_and_oar_lambda():
    from oar import get_default_layer_options_from_options

    options = {
        "oar": {"omega": 6, "oar_lambda": 1e-4},
        "s": 1.0,
        "t": 1.0,
    }
    layer_options = get_default_layer_options_from_options(options)
    assert layer_options["QRNN_0"]["oar"]["omega"] == 6
    assert layer_options["QRNN_0"]["oar"]["oar_lambda"] == 1e-4

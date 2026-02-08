import tempfile

import tensorflow as tf

from oar import (
    QDenseWithOAR,
    QRNNWithOAR,
    QSimpleRNNCellWithOAR,
    ReservoirHistogramCallback,
    _reservoir_update,
    reset_stat_weights,
)


def test_reservoir_update_fills_initial_slots():
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
    tf.debugging.assert_equal(updated[:4], new_vals)


def test_reservoir_update_replaces_after_full():
    reservoir = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
    count = tf.constant(5, dtype=tf.int64)
    new_vals = tf.constant([99.0, 99.0, 99.0], dtype=tf.float32)

    updated, updated_count = _reservoir_update(
        reservoir=reservoir,
        count=count,
        new_values=new_vals,
        reservoir_size=5,
        seed=42,
    )

    tf.debugging.assert_equal(updated_count, tf.constant(8, dtype=tf.int64))
    # Can't assert exact values due to randomness, but shape should be preserved
    assert updated.shape.as_list() == [5]


def test_qsimple_rnn_cell_has_reservoir_weights_default():
    cell = QSimpleRNNCellWithOAR(units=4, batch_size=2, name="QRNN_0")
    cell.build(tf.TensorShape([2, 8]))
    assert cell.preact_reservoir.shape.as_list() == [100000]
    assert cell.reservoir_count.dtype == tf.int64


def test_qsimple_rnn_cell_has_reservoir_weights_custom():
    cell = QSimpleRNNCellWithOAR(
        units=4, batch_size=2, reservoir_size=500, name="QRNN_0"
    )
    cell.build(tf.TensorShape([2, 8]))
    assert cell.preact_reservoir.shape.as_list() == [500]


def test_qdense_has_reservoir_weights_default():
    layer = QDenseWithOAR(units=3, batch_size=2, name="DENSE_0")
    layer.build(tf.TensorShape([2, 8]))
    assert layer.preact_reservoir.shape.as_list() == [100000]
    assert layer.reservoir_count.dtype == tf.int64


def test_qdense_has_reservoir_weights_custom():
    layer = QDenseWithOAR(units=3, batch_size=2, reservoir_size=1000, name="DENSE_0")
    layer.build(tf.TensorShape([2, 8]))
    assert layer.preact_reservoir.shape.as_list() == [1000]


def test_qrnn_with_oar_propagates_reservoir_size():
    layer = QRNNWithOAR(units=4, batch_size=2, reservoir_size=777, name="QRNN_0")
    layer.build(tf.TensorShape([2, 10, 8]))
    assert layer.cell.preact_reservoir.shape.as_list() == [777]


def test_reservoir_histogram_callback_constructs():
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = ReservoirHistogramCallback(log_dir=tmpdir)
        assert cb is not None
        assert cb.log_dir == tmpdir


def test_reset_stat_weights_zeros_reservoir():
    class DummyModel:
        def __init__(self):
            self.weights = [
                tf.Variable([1.0, 2.0, 3.0], name="layer/preact_reservoir:0"),
                tf.Variable([5], dtype=tf.int64, name="layer/reservoir_count:0"),
                tf.Variable([9.0], name="layer/kernel:0"),
            ]

        def get_weights(self):
            return [w.numpy() for w in self.weights]

        def set_weights(self, new_weights):
            for w, nw in zip(self.weights, new_weights):
                w.assign(nw)

    model = DummyModel()
    reset_stat_weights(model)
    tf.debugging.assert_equal(model.weights[0], tf.zeros([3]))
    tf.debugging.assert_equal(model.weights[1], tf.zeros([], dtype=tf.int64))
    tf.debugging.assert_equal(model.weights[2], tf.constant([9.0]))

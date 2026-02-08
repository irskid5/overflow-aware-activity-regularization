"""OAR training callbacks and utilities."""

import numpy as np
import tensorflow as tf


# Update reservoir every N batches to reduce overhead.
RESERVOIR_UPDATE_EVERY = 20


def _reservoir_update(reservoir, count, new_values, reservoir_size, seed):
    """
    Vitter's Algorithm R for reservoir sampling (NumPy backend).
    
    Maintains a fixed-size reservoir of samples from a stream,
    ensuring each element has equal probability of being included.
    
    Uses NumPy via tf.numpy_function to avoid slow TF scatter ops in TF 2.10.
    
    Args:
        reservoir: Current reservoir tensor [reservoir_size]
        count: Total samples seen so far (int64 scalar)
        new_values: New values to add (any shape, will be flattened)
        reservoir_size: Maximum reservoir size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (updated_reservoir, updated_count)
    """
    def _numpy_reservoir_update(
        reservoir_np, count_np, new_values_np, reservoir_size_np, seed_np
    ):
        flat = new_values_np.reshape(-1)
        reservoir_size_i = int(reservoir_size_np)
        count_i = int(count_np)

        # Phase 1: Fill initial slots
        capacity_left = reservoir_size_i - count_i
        fill_n = max(0, min(flat.size, capacity_left))
        if fill_n > 0:
            reservoir_np = reservoir_np.copy()
            reservoir_np[count_i : count_i + fill_n] = flat[:fill_n]
            count_i += fill_n

        # Phase 2: Reservoir sampling
        remaining = flat[fill_n:]
        if remaining.size > 0:
            rng = np.random.default_rng(int(seed_np) + count_i)
            i_range = np.arange(remaining.size, dtype=np.int64)
            total_idx = count_i + i_range + 1
            accept_threshold = reservoir_size_i / total_idx.astype(np.float64)
            accept_mask = rng.random(remaining.size) < accept_threshold
            accepted_idx = np.where(accept_mask)[0]
            if accepted_idx.size > 0:
                slots = rng.integers(
                    low=0, high=reservoir_size_i, size=accepted_idx.size
                )
                reservoir_np = reservoir_np.copy()
                reservoir_np[slots] = remaining[accepted_idx]
            count_i += remaining.size

        return reservoir_np, np.array(count_i, dtype=np.int64)

    updated_reservoir, updated_count = tf.numpy_function(
        _numpy_reservoir_update,
        [reservoir, count, new_values, reservoir_size, seed],
        [reservoir.dtype, tf.int64],
        name="reservoir_update_np",
    )
    updated_reservoir.set_shape(reservoir.shape)
    updated_count.set_shape(count.shape)
    return updated_reservoir, updated_count


def reset_stat_weights(model):
    """
    Zero out stat-tracking weights in a model.
    
    Resets reservoir counts, EMA statistics, and other tracking weights
    that should not persist between training steps.
    
    Args:
        model: Keras model with stat-tracking weights
    """
    weights = model.get_weights()
    for i in range(len(weights)):
        name = model.weights[i].name
        if any(
            pattern in name
            for pattern in [
                "/w",
                "/x",
                "preact_reservoir",
                "reservoir_count",
                "reservoir_step",
            ]
        ):
            weights[i] = 0 * weights[i]
    model.set_weights(weights)


class ReservoirHistogramCallback(tf.keras.callbacks.Callback):
    """
    Logs preact_reservoir histograms to TensorBoard at epoch end.
    
    Scans model weights for reservoir buffers and logs their distributions
    as histograms, enabling visualization of pre-activation distributions.
    
    Args:
        log_dir: TensorBoard log directory
        reservoir_weight_name: Pattern to match reservoir weight names
    """

    def __init__(self, log_dir, reservoir_weight_name="preact_reservoir"):
        super().__init__()
        self.log_dir = log_dir
        self.reservoir_weight_name = reservoir_weight_name
        self.writer = None

    def set_model(self, model):
        super().set_model(model)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if self.writer is None:
            return
        with self.writer.as_default():
            for weight in self.model.weights:
                if self.reservoir_weight_name in weight.name:
                    tf.summary.histogram(weight.name, weight, step=epoch)
            self.writer.flush()

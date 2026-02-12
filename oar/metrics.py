# oar/metrics.py
"""Custom metrics for OAR models."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="OAR")
class Perplexity(tf.keras.metrics.Metric):
    """Perplexity metric for language modeling.

    Perplexity = exp(average cross-entropy loss)

    Lower is better. A perplexity of N means the model is as uncertain
    as if it were choosing uniformly among N words.
    """

    def __init__(self, name: str = "perplexity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate cross-entropy loss.

        Args:
            y_true: (batch, seq_len) int32 target word IDs
            y_pred: (batch, seq_len, vocab_size) float32 probabilities
            sample_weight: Optional weights (unused)
        """
        # Compute per-token cross-entropy
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        self.total_loss.assign_add(tf.reduce_sum(loss))
        self.count.assign_add(tf.cast(tf.size(loss), tf.float32))

    def result(self):
        """Return perplexity = exp(average loss)."""
        return tf.exp(self.total_loss / tf.maximum(self.count, 1e-7))

    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        return config

    def reset_state(self):
        """Reset accumulated loss and count."""
        self.total_loss.assign(0.0)
        self.count.assign(0.0)

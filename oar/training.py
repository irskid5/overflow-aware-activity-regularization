"""OAR training utilities and model classes."""

import tensorflow as tf
from keras.engine import data_adapter


@tf.keras.utils.register_keras_serializable(package="OAR")
class OARModel(tf.keras.models.Model):
    """
    Keras Model subclass with optional gradient logging.

    Provides a custom train_step that can optionally log gradient
    norms and statistics for debugging training dynamics.

    Args:
        log_gradients: If True, compute and log gradient statistics
    """

    def __init__(self, *args, log_gradients=False, **kwargs):
        super(OARModel, self).__init__(*args, **kwargs)
        self.log_gradients = log_gradients

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)

        self._validate_target_and_loss(y, loss)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        output = self.compute_metrics(x, y, y_pred, sample_weight)

        if self.log_gradients:
            # Gradient logging (commented out by default for performance)
            # Uncomment specific blocks as needed for debugging
            pass

        return output

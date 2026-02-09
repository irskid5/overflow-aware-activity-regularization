"""MNIST RNN model architecture with OAR support."""

import tensorflow as tf
from qkeras import *

from oar import (
    Downsampling,
    QDenseWithOAR,
    QRNNWithOAR,
    OARModel,
    TrackedActivation,
    TernarizationWithThreshold,
    ternarize_tensor_with_threshold,
)
from oar.config import TrainingStepConfig, LayerConfig, resolve_activation
from experiments.mnist.steps import get_default_layer_config

SEED = 1997

# Regularizers
kernel_regularizer = None
recurrent_regularizer = None
bias_regularizer = None
activation_regularizer = None

# Initializers
rnn_kernel_initializer = tf.keras.initializers.VarianceScaling(
    scale=1.0, mode="fan_avg", distribution="uniform", seed=SEED
)
rnn_recurrent_initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=SEED)
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_in", distribution="truncated_normal", seed=SEED
)


def get_model(step_config: TrainingStepConfig) -> OARModel:
    """Build MNIST RNN model from step configuration.

    Args:
        step_config: Training step configuration

    Returns:
        Compiled OARModel
    """

    def get_layer_cfg(name: str) -> LayerConfig:
        """Get config for layer, using defaults if not specified."""
        return step_config.layers.get(name, get_default_layer_config(name))

    def make_quantizer(threshold: float | None, name: str = None):
        """Create quantizer if threshold is set."""
        if threshold is None:
            return None
        return TernarizationWithThreshold(threshold=threshold, name=name)

    # Input shape based on enlarge setting
    if step_config.enlarge:
        input_shape = (128, 128, 1)
        seq_len, features = 128, 128
        model_name = "ENLARGED_MNIST_RNN"
    else:
        input_shape = (28, 28, 1)
        seq_len, features = 28, 28
        model_name = "MNIST_RNN"

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Input ternarization
    if step_config.input_config.quantize_threshold is not None:
        theta = step_config.input_config.quantize_threshold
        x = tf.keras.layers.Lambda(
            lambda x: tf.stop_gradient(
                ternarize_tensor_with_threshold(
                    x, theta=theta * tf.reduce_mean(tf.abs(x))
                )
            ),
            trainable=False,
            dtype=tf.float32,
            name="TERNARIZE_WITH_THRESHOLD",
        )(inputs)
    else:
        x = tf.keras.layers.Lambda(lambda x: x, name="NOOP")(inputs)

    x = tf.keras.layers.Reshape((seq_len, features))(x)

    # QRNN_0
    cfg = get_layer_cfg("QRNN_0")
    x = QRNNWithOAR(
        cell=None,
        units=128,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation, cfg.omega),
            name="QRNN_0",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        return_sequences=True,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=make_quantizer(cfg.quantize_threshold, "QRNN_0/quantized_kernel"),
        recurrent_quantizer=make_quantizer(cfg.quantize_threshold, "QRNN_0/quantized_recurrent"),
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        use_oar=cfg.oar_lambda is not None,
        oar_lambda=cfg.oar_lambda or 0.0,
        omega=cfg.omega,
        s=cfg.gradient_scale,
        name="QRNN_0",
    )(x)

    x = Downsampling(reduction_factor=2)(x)

    # QRNN_1
    cfg = get_layer_cfg("QRNN_1")
    x = QRNNWithOAR(
        cell=None,
        units=128,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation, cfg.omega),
            name="QRNN_1",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        return_sequences=True,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=make_quantizer(cfg.quantize_threshold, "QRNN_1/quantized_kernel"),
        recurrent_quantizer=make_quantizer(cfg.quantize_threshold, "QRNN_1/quantized_recurrent"),
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        use_oar=cfg.oar_lambda is not None,
        oar_lambda=cfg.oar_lambda or 0.0,
        omega=cfg.omega,
        s=cfg.gradient_scale,
        name="QRNN_1",
    )(x)

    x = tf.keras.layers.Flatten()(x)

    # DENSE_0
    cfg = get_layer_cfg("DENSE_0")
    x = QDenseWithOAR(
        units=1024,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation, cfg.omega),
            name="DENSE_0",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        kernel_regularizer=kernel_regularizer,
        kernel_quantizer=make_quantizer(cfg.quantize_threshold, "DENSE_0"),
        kernel_initializer=dense_kernel_initializer,
        use_oar=cfg.oar_lambda is not None,
        oar_lambda=cfg.oar_lambda or 0.0,
        omega=cfg.omega,
        s=cfg.gradient_scale,
        name="DENSE_0",
    )(x)

    # DENSE_OUT
    cfg = get_layer_cfg("DENSE_OUT")
    outputs = QDenseWithOAR(
        units=10,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation, cfg.omega),
            name="DENSE_OUT",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        kernel_regularizer=kernel_regularizer,
        kernel_quantizer=make_quantizer(cfg.quantize_threshold, "DENSE_OUT"),
        kernel_initializer=dense_kernel_initializer,
        use_oar=cfg.oar_lambda is not None,
        oar_lambda=cfg.oar_lambda or 0.0,
        omega=cfg.omega,
        s=cfg.gradient_scale,
        name="DENSE_OUT",
    )(x)

    model = OARModel(
        inputs=[inputs],
        outputs=[outputs],
        name=model_name,
    )

    model.summary()

    return model

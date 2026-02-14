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
    resolve_activation,
)
from oar.config import StepConfig, LayerStepConfig

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


def get_model(
    step_config: StepConfig,
    pretrained_weights: str | None = None,
) -> OARModel:
    """Build MNIST RNN model from step configuration.
    
    If pretrained_weights is provided and any layer needs computed thresholds
    (ternarization_scale set but no explicit threshold), this function:
    1. Builds a temporary model
    2. Loads the pretrained weights
    3. Computes thresholds as τ = ternarization_scale × E[|θ|]
    4. Builds the final model with computed thresholds
    
    Args:
        step_config: Training step configuration
        pretrained_weights: Path to checkpoint (for threshold computation)
        
    Returns:
        Compiled OARModel ready for training
    """
    # Check if we need to compute thresholds
    thresholds = {}
    if pretrained_weights:
        thresholds = _compute_thresholds_if_needed(step_config, pretrained_weights)
    
    # Build model with thresholds
    model = _build_model(step_config, thresholds)
    
    # Load weights if provided
    if pretrained_weights:
        model.load_weights(pretrained_weights)
        print(f"Restored pretrained weights from {pretrained_weights}.")
    
    return model


def _compute_thresholds_if_needed(
    step_config: StepConfig,
    checkpoint_path: str,
) -> dict[str, float]:
    """Compute thresholds: τ = ternarization_scale × E[|θ|].
    
    Only computes for layers that have ternarization_scale set but no explicit threshold.
    Skips INPUT layer (its threshold is used directly, not computed from weights).
    """
    # First check if any layer needs computed thresholds
    layers_needing_thresholds = []
    for name, cfg in step_config.layers.items():
        q = cfg.quantization
        if q.ternarization_scale is not None and q.threshold is None and name != "INPUT":
            layers_needing_thresholds.append(name)
    
    if not layers_needing_thresholds:
        return {}
    
    # Build temp model and load weights
    temp_model = _build_model(step_config, thresholds={})
    temp_model.load_weights(checkpoint_path)
    
    # Compute thresholds
    thresholds = {}
    for layer_name in layers_needing_thresholds:
        t = step_config.layers[layer_name].quantization.ternarization_scale
        
        for layer in temp_model.layers:
            if layer_name in layer.name and layer.trainable_weights:
                all_weights = tf.concat(
                    [tf.reshape(w, [-1]) for w in layer.trainable_weights], axis=-1
                )
                mean_abs = float(tf.reduce_mean(tf.abs(all_weights)).numpy())
                thresholds[layer_name] = t * mean_abs
                break
    
    print(f"Computed thresholds: {thresholds}")
    return thresholds


def _get_default_layer_config() -> LayerStepConfig:
    """Return default layer config (tanh activation, no quantization)."""
    return LayerStepConfig()


def _build_model(step_config: StepConfig, thresholds: dict[str, float]) -> OARModel:
    """Build the actual model architecture.
    
    Args:
        step_config: Training step configuration
        thresholds: Pre-computed thresholds for layers (layer_name -> threshold)
        
    Returns:
        OARModel instance with configured layers
    """
    def get_layer_cfg(name: str) -> LayerStepConfig:
        """Get config for layer, using defaults if not specified."""
        return step_config.layers.get(name, _get_default_layer_config())
    
    def get_threshold(name: str, cfg: LayerStepConfig) -> float | None:
        """Get threshold for a layer, using computed or explicit."""
        if name in thresholds:
            return thresholds[name]
        return cfg.quantization.threshold
    
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

    # Input ternarization (uses ternarization_scale directly as threshold)
    input_cfg = get_layer_cfg("INPUT")
    input_threshold = input_cfg.quantization.ternarization_scale
    if input_threshold is not None:
        x = tf.keras.layers.Lambda(
            lambda x, theta=input_threshold: tf.stop_gradient(
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
    threshold = get_threshold("QRNN_0", cfg)
    oar_cfg = cfg.quantization.oar
    x = QRNNWithOAR(
        cell=None,
        units=128,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation),
            name="QRNN_0",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        return_sequences=True,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=make_quantizer(threshold, "QRNN_0/quantized_kernel"),
        recurrent_quantizer=make_quantizer(threshold, "QRNN_0/quantized_recurrent"),
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        use_oar=oar_cfg is not None,
        oar_lambda=oar_cfg.regularization_rate if oar_cfg else 0.0,
        omega=oar_cfg.omega if oar_cfg else 6,
        s=cfg.activation.gradient_scale or 1.0,
        name="QRNN_0",
    )(x)

    x = Downsampling(reduction_factor=2)(x)

    # QRNN_1
    cfg = get_layer_cfg("QRNN_1")
    threshold = get_threshold("QRNN_1", cfg)
    oar_cfg = cfg.quantization.oar
    x = QRNNWithOAR(
        cell=None,
        units=128,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation),
            name="QRNN_1",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        return_sequences=True,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=make_quantizer(threshold, "QRNN_1/quantized_kernel"),
        recurrent_quantizer=make_quantizer(threshold, "QRNN_1/quantized_recurrent"),
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        use_oar=oar_cfg is not None,
        oar_lambda=oar_cfg.regularization_rate if oar_cfg else 0.0,
        omega=oar_cfg.omega if oar_cfg else 6,
        s=cfg.activation.gradient_scale or 1.0,
        name="QRNN_1",
    )(x)

    x = tf.keras.layers.Flatten()(x)

    # DENSE_0
    cfg = get_layer_cfg("DENSE_0")
    threshold = get_threshold("DENSE_0", cfg)
    oar_cfg = cfg.quantization.oar
    x = QDenseWithOAR(
        units=1024,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation),
            name="DENSE_0",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        kernel_regularizer=kernel_regularizer,
        kernel_quantizer=make_quantizer(threshold, "DENSE_0"),
        kernel_initializer=dense_kernel_initializer,
        use_oar=oar_cfg is not None,
        oar_lambda=oar_cfg.regularization_rate if oar_cfg else 0.0,
        omega=oar_cfg.omega if oar_cfg else 6,
        s=cfg.activation.gradient_scale or 1.0,
        name="DENSE_0",
    )(x)

    # DENSE_OUT
    cfg = get_layer_cfg("DENSE_OUT")
    threshold = get_threshold("DENSE_OUT", cfg)
    oar_cfg = cfg.quantization.oar
    outputs = QDenseWithOAR(
        units=10,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation),
            name="DENSE_OUT",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        kernel_regularizer=kernel_regularizer,
        kernel_quantizer=make_quantizer(threshold, "DENSE_OUT"),
        kernel_initializer=dense_kernel_initializer,
        use_oar=oar_cfg is not None,
        oar_lambda=oar_cfg.regularization_rate if oar_cfg else 0.0,
        omega=oar_cfg.omega if oar_cfg else 6,
        s=cfg.activation.gradient_scale or 1.0,
        name="DENSE_OUT",
    )(x)

    model = OARModel(
        inputs=[inputs],
        outputs=[outputs],
        name=model_name,
    )

    model.summary()

    return model

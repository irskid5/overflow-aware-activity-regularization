"""Speech Commands RNN model architecture with OAR support.

Same architecture as MNIST but with different input dimensions:
- Input: (NUM_FRAMES, NUM_MFCC) instead of (28, 28)
- Output: NUM_CLASSES (12) instead of 10
"""
import tensorflow as tf

from experiments.speech_commands.data import NUM_CLASSES, NUM_FRAMES, NUM_MFCC
from oar import (
    Downsampling,
    QDenseWithOAR,
    QRNNWithOAR,
    OARModel,
    TrackedActivation,
    TernarizationWithThreshold,
    ternarize_tensor_with_threshold,
    resolve_activation,
    compute_thresholds_if_needed,
)
from oar.config import StepConfig, LayerStepConfig

SEED = 1997

# Hidden dimensions (similar to MNIST)
RNN_UNITS = 128
DENSE_UNITS = 512  # Reduced from MNIST's 1024 for this task

# Initializers
rnn_kernel_initializer = tf.keras.initializers.VarianceScaling(
    scale=1.0, mode="fan_avg", distribution="uniform", seed=SEED
)
rnn_recurrent_initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=SEED)
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_in", distribution="truncated_normal", seed=SEED
)

# Regularizers (None by default, can be set for weight decay)
kernel_regularizer = None
recurrent_regularizer = None


def get_model(
    step_config: StepConfig,
    pretrained_weights: str | None = None,
) -> OARModel:
    """Build Speech Commands model from step configuration.
    
    Args:
        step_config: Training step configuration
        pretrained_weights: Path to checkpoint for threshold computation
        
    Returns:
        OARModel ready for training
    """
    thresholds = {}
    if pretrained_weights:
        thresholds = compute_thresholds_if_needed(
            step_config, pretrained_weights, _build_model
        )
    
    model = _build_model(step_config, thresholds)
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
        print(f"Restored pretrained weights from {pretrained_weights}.")
    
    return model


def _get_default_layer_config() -> LayerStepConfig:
    """Return default layer config."""
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
        return step_config.layers.get(name, _get_default_layer_config())
    
    def get_threshold(name: str, cfg: LayerStepConfig) -> float | None:
        if name in thresholds:
            return thresholds[name]
        return cfg.quantization.threshold
    
    def make_quantizer(threshold: float | None, name: str = None):
        if threshold is None:
            return None
        return TernarizationWithThreshold(threshold=threshold, name=name)

    # Input: (frames, mfcc_coeffs)
    inputs = tf.keras.layers.Input(shape=(NUM_FRAMES, NUM_MFCC))

    # Input ternarization
    input_cfg = get_layer_cfg("INPUT")
    input_threshold = input_cfg.quantization.ternarization_scale
    if input_threshold is not None:
        x = tf.keras.layers.Lambda(
            lambda x, theta=input_threshold: tf.stop_gradient(
                ternarize_tensor_with_threshold(
                    # Add epsilon to handle silence (all-zero) inputs
                    x, theta=tf.maximum(theta * tf.reduce_mean(tf.abs(x)), 1e-6)
                )
            ),
            trainable=False,
            dtype=tf.float32,
            name="TERNARIZE_WITH_THRESHOLD",
        )(inputs)
    else:
        x = tf.keras.layers.Lambda(lambda x: x, name="NOOP")(inputs)

    # QRNN_0
    cfg = get_layer_cfg("QRNN_0")
    threshold = get_threshold("QRNN_0", cfg)
    oar_cfg = cfg.quantization.oar
    x = QRNNWithOAR(
        cell=None,
        units=RNN_UNITS,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation),
            name="QRNN_0",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        return_sequences=True,
        kernel_quantizer=make_quantizer(threshold, "QRNN_0/quantized_kernel"),
        recurrent_quantizer=make_quantizer(threshold, "QRNN_0/quantized_recurrent"),
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
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
        units=RNN_UNITS,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation),
            name="QRNN_1",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        return_sequences=True,
        kernel_quantizer=make_quantizer(threshold, "QRNN_1/quantized_kernel"),
        recurrent_quantizer=make_quantizer(threshold, "QRNN_1/quantized_recurrent"),
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
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
        units=DENSE_UNITS,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation),
            name="DENSE_0",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        kernel_quantizer=make_quantizer(threshold, "DENSE_0"),
        kernel_initializer=dense_kernel_initializer,
        kernel_regularizer=kernel_regularizer,
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
        units=NUM_CLASSES,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation),
            name="DENSE_OUT",
        ),
        batch_size=step_config.batch_size,
        use_bias=False,
        kernel_quantizer=make_quantizer(threshold, "DENSE_OUT"),
        kernel_initializer=dense_kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_oar=oar_cfg is not None,
        oar_lambda=oar_cfg.regularization_rate if oar_cfg else 0.0,
        omega=oar_cfg.omega if oar_cfg else 6,
        s=cfg.activation.gradient_scale or 1.0,
        name="DENSE_OUT",
    )(x)

    model = OARModel(
        inputs=[inputs],
        outputs=[outputs],
        name="SPEECH_COMMANDS_RNN",
    )

    model.summary()
    return model

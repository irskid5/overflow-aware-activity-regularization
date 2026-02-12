"""PTB language model builder.

Architecture: Embedding → QRNN_0 → QRNN_1 → DENSE_OUT
- Stateful RNN for language modeling
- No Downsampling or DENSE_0 (unlike MNIST)
- Embedding output ternarization (not weights)

Key design: Provides make_model_factory(vocab_size) to create a
ModelFactory-compatible callable for use with oar.runner.Runner.
"""
import os
from functools import partial
from typing import Callable

import tensorflow as tf

from experiments.mnist.model import resolve_activation  # Reuse from MNIST
from experiments.ptb.config import BATCH_SIZE, EMBED_DIM, NUM_STEPS
from oar import (
    OARModel,
    QDenseWithOAR,
    QRNNWithOAR,
    TernarizationWithThreshold,
    TrackedActivation,
    ternarize_tensor_with_threshold,
)
from oar.config import LayerStepConfig, StepConfig


def _get_layer_config(step_config: StepConfig, layer_name: str) -> LayerStepConfig:
    """Get layer config, defaulting to empty config if not present."""
    return step_config.layers.get(layer_name, LayerStepConfig())


def _build_qrnn(
    x: tf.Tensor,
    name: str,
    step_config: StepConfig,
    thresholds: dict[str, float],
) -> tf.Tensor:
    """Build a QRNN layer with OAR support.
    
    Args:
        x: Input tensor
        name: Layer name (e.g., "QRNN_0")
        step_config: Training step configuration
        thresholds: Pre-computed thresholds
        
    Returns:
        Output tensor
    """
    cfg = _get_layer_config(step_config, name)
    activation = resolve_activation(cfg.activation)

    use_oar = cfg.quantization.oar is not None
    oar_lambda = cfg.quantization.oar.regularization_rate if use_oar else 0
    omega = cfg.quantization.oar.omega if use_oar else 32

    kernel_quantizer = None
    if cfg.quantization.ternarization_scale is not None:
        threshold = thresholds.get(name, 0.5)
        kernel_quantizer = TernarizationWithThreshold(threshold=threshold)

    return QRNNWithOAR(
        units=EMBED_DIM,
        activation=TrackedActivation(activation, name=name),
        batch_size=BATCH_SIZE,
        stateful=True,
        return_sequences=True,
        kernel_quantizer=kernel_quantizer,
        recurrent_quantizer=kernel_quantizer,
        use_oar=use_oar,
        oar_lambda=oar_lambda,
        omega=omega,
        name=name,
    )(x)


def _build_model(
    step_config: StepConfig,
    vocab_size: int,
    thresholds: dict[str, float],
) -> OARModel:
    """Build PTB model from step config.

    Args:
        step_config: Training step configuration
        vocab_size: Vocabulary size for embedding and output
        thresholds: Pre-computed ternarization thresholds

    Returns:
        OARModel instance
    """
    # Stateful RNN requires explicit batch_shape
    inputs = tf.keras.Input(
        batch_shape=(BATCH_SIZE, NUM_STEPS),
        dtype=tf.int32,
        name="word_ids",
    )

    # Embedding layer (trainable, full precision weights)
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        name="EMBEDDING",
    )(inputs)

    # Embedding output ternarization (Step 3+)
    input_cfg = _get_layer_config(step_config, "INPUT")
    if input_cfg.quantization.ternarization_scale is not None:
        scale = input_cfg.quantization.ternarization_scale

        def ternarize_fn(x, scale=scale):
            theta = scale * tf.reduce_mean(tf.abs(x))
            return ternarize_tensor_with_threshold(x, theta=theta)

        x = tf.keras.layers.Lambda(ternarize_fn, name="TERNARIZE_EMBEDDING")(x)

    # QRNN layers (stateful, reuse helper to avoid duplication)
    x = _build_qrnn(x, "QRNN_0", step_config, thresholds)
    x = _build_qrnn(x, "QRNN_1", step_config, thresholds)

    # DENSE_OUT (vocab_size outputs per timestep)
    cfg = _get_layer_config(step_config, "DENSE_OUT")
    activation = resolve_activation(cfg.activation)

    kernel_quantizer = None
    if cfg.quantization.ternarization_scale is not None:
        threshold = thresholds.get("DENSE_OUT", 0.5)
        kernel_quantizer = TernarizationWithThreshold(threshold=threshold)

    outputs = QDenseWithOAR(
        units=vocab_size,
        activation=TrackedActivation(activation, name="DENSE_OUT"),
        batch_size=BATCH_SIZE,
        kernel_quantizer=kernel_quantizer,
        bias_quantizer=kernel_quantizer,
        use_oar=False,  # No OAR on output layer
        name="DENSE_OUT",
    )(x)

    return OARModel(inputs=[inputs], outputs=[outputs], name="PTB_RNN")


def _compute_thresholds(
    step_config: StepConfig,
    model: tf.keras.Model,
) -> dict[str, float]:
    """Compute ternarization thresholds from model weights.

    Threshold τ = ternarization_scale × mean(|weights|)

    Args:
        step_config: Training step configuration
        model: Model with trained weights

    Returns:
        Dict mapping layer name to threshold
    """
    thresholds = {}

    for layer_name in ["QRNN_0", "QRNN_1", "DENSE_OUT"]:
        cfg = _get_layer_config(step_config, layer_name)
        if cfg.quantization.ternarization_scale is None:
            continue

        scale = cfg.quantization.ternarization_scale

        # Find layer and get kernel weights
        try:
            layer = model.get_layer(layer_name)
            if hasattr(layer, "cell"):
                # RNN layer - get cell kernel
                kernel = layer.cell.kernel
            else:
                kernel = layer.kernel
            threshold = scale * tf.reduce_mean(tf.abs(kernel)).numpy()
            thresholds[layer_name] = float(threshold)
            print(f"Computed threshold for {layer_name}: {threshold:.4f}")
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not compute threshold for {layer_name}: {e}")

    return thresholds


def _get_model(
    step_config: StepConfig,
    vocab_size: int,
    pretrained_weights: str | None = None,
) -> OARModel:
    """Build PTB model for given training step.

    Args:
        step_config: Training step configuration
        vocab_size: Vocabulary size for embedding and output
        pretrained_weights: Optional path to pretrained weights

    Returns:
        OARModel ready for training
        
    Raises:
        FileNotFoundError: If pretrained_weights path doesn't exist
    """
    # Validate pretrained weights path
    if pretrained_weights is not None and not os.path.exists(pretrained_weights):
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_weights}")
    
    # Compute thresholds from pretrained weights if needed
    thresholds = {}
    if pretrained_weights is not None:
        temp_model = _build_model(step_config, vocab_size, thresholds={})
        temp_model.load_weights(pretrained_weights)
        thresholds = _compute_thresholds(step_config, temp_model)
        del temp_model

    # Build final model with computed thresholds
    model = _build_model(step_config, vocab_size, thresholds)

    # Load pretrained weights
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print(f"Loaded pretrained weights from {pretrained_weights}")

    return model


def make_model_factory(vocab_size: int) -> Callable:
    """Create a ModelFactory-compatible callable with vocab_size bound.
    
    This allows using oar.runner.Runner without modification.
    
    Args:
        vocab_size: Vocabulary size for embedding and output
        
    Returns:
        Callable matching ModelFactory protocol:
        (step_config, pretrained_weights=None) -> Model
        
    Example:
        vocab_size = 10000
        model_factory = make_model_factory(vocab_size)
        runner = Runner(experiment, model_factory, data_loader)
    """
    return partial(_get_model, vocab_size=vocab_size)
